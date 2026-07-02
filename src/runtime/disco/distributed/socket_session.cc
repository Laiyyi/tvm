/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>

#include <numeric>
#include <thread>

#include "../../../support/socket.h"
#include "../bcast_session.h"
#include "../message_queue.h"


namespace tvm {
namespace runtime {

using namespace tvm::support;

enum class DiscoSocketAction {
  kShutdown = static_cast<int>(DiscoAction::kShutDown),
  kSend,
  kReceive,
};

class DiscoSocketChannel : public DiscoChannel {
 public:
  explicit DiscoSocketChannel(const TCPSocket& socket)
      : socket_(socket), message_queue_(&socket_) {}

  DiscoSocketChannel(DiscoSocketChannel&& other) = delete;
  DiscoSocketChannel(const DiscoSocketChannel& other) = delete;
  void Send(const ffi::PackedArgs& args) { message_queue_.Send(args); }
  ffi::PackedArgs Recv() { return message_queue_.Recv(); }
  void Reply(const ffi::PackedArgs& args) { message_queue_.Send(args); }
  ffi::PackedArgs RecvReply() { return message_queue_.Recv(); }

 private:
  TCPSocket socket_;
  DiscoStreamMessageQueue message_queue_;
};

// Proxy thread helpers — both controller and remote share these.
// send: pipe → TCP
// recv: TCP → pipe
constexpr size_t kProxyBufSize = 64 * 1024;

void ProxyLoop(DiscoRingChannel* src, DiscoRingChannel* dst, std::string tag) {

  LOG(INFO) << "[Proxy " << tag << "] entering, src=" << src << " dst=" << dst;
  std::vector<char> buf(kProxyBufSize);

  while (true) {
    ssize_t n = src->ReadSome(buf.data(), buf.size());
    if (n <= 0) { LOG(INFO) << "[Proxy " << tag << "] src closed (n=" << n << "), exit"; return; }
    LOG(INFO) << "[Proxy " << tag << "] transfer " << n << " bytes";

    ssize_t written = 0;
    while (written < n) {
      ssize_t w = dst->WriteSome(buf.data() + written, n - written);
      if (w <= 0) { LOG(INFO) << "[Proxy " << tag << "] dst failed (w=" << w << "), exit"; return; }
      written += w;
    }
  }
}

class SocketSessionObj : public BcastSessionObj {
 public:
  explicit SocketSessionObj(int num_nodes, int num_workers_per_node, int num_groups,
                            const ffi::String& host, int port,
                            bool build_ring)
      : num_nodes_(num_nodes), num_workers_per_node_(num_workers_per_node) {
    const auto f_create_local_session =
        tvm::ffi::Function::GetGlobal("runtime.disco.create_socket_session_local_workers");
    ICHECK(f_create_local_session.has_value())
        << "Cannot find function runtime.disco.create_socket_session_local_workers";
    local_session_ = ((*f_create_local_session)(num_workers_per_node, build_ring)).cast<BcastSession>();
    DRef f_init_workers =
        local_session_->GetGlobalFunc("runtime.disco.socket_session_init_workers");
    local_session_->CallPacked(f_init_workers, num_nodes_, /*node_id=*/0, num_groups,
                               num_workers_per_node_);

    Socket::Startup();
    socket_.Create();
    socket_.SetKeepAlive(true);
    socket_.Bind(SockAddr(host.c_str(), port));
    socket_.Listen();
    LOG(INFO) << "[Host] SocketSession controller listening on " << host << ":" << port;

    AnyView packed_args[5];
    packed_args[0] = num_nodes;
    packed_args[1] = num_workers_per_node;
    packed_args[2] = num_groups;
    packed_args[4] = build_ring;

    if (build_ring) {
      node_hosts_.reserve(num_nodes);
      node_hosts_.push_back(host);
    }
 
    for (int i = 0; i + 1 < num_nodes; ++i) {
      SockAddr addr;
      remote_sockets_.push_back(socket_.Accept(&addr));
      remote_channels_.emplace_back(std::make_unique<DiscoSocketChannel>(remote_sockets_.back()));
      packed_args[3] = i + 1;
      // Send metadata to each remote node:
      //  - num_nodes
      //  - num_workers_per_node
      //  - num_groups
      //  - node_id
      //  - build_ring
      remote_channels_.back()->Send(ffi::PackedArgs(packed_args, 5));
      LOG(INFO) << "[Host] Remote node " << addr.AsString() << " connected";

      if (build_ring) {
        std::string full = addr.AsString();
        ffi::String ip(full.substr(0, full.find(':')));
        node_hosts_.push_back(ip);
        LOG(INFO) << "[Host] IP table : node " << (i + 1) << " ip=" << ip.c_str(); 
      }
    }


    if( build_ring && num_nodes_ > 1) {

        const int base_ring_port  = port + 1;
        const int socket_node_id  = 0;
        const int next_node_id    = (socket_node_id + 1) % num_nodes_;
        const int recv_port       = base_ring_port + socket_node_id;
        const int send_port       = base_ring_port + next_node_id;

        LOG(INFO) << "[Host] RingTcp controller setup: socket_node=" << socket_node_id
            << " listen=:" << recv_port
            << " connect_to=" << node_hosts_[next_node_id].c_str() << ":" << send_port;

        //Broadcast ip and base_ring_port to all remote workers.
        {
          std::vector<AnyView> baseport_ip(num_nodes_ + 1);
          baseport_ip[0] = base_ring_port;
          for (int i = 0; i < num_nodes_; ++i) baseport_ip[1 + i] = node_hosts_[i];
          for (auto& ch : remote_channels_) {
             ch->Send(ffi::PackedArgs(baseport_ip.data(), baseport_ip.size()));
          }
          LOG(INFO) << "[Host] IP and port broadcast completed.";
        }

        TCPSocket recv_sock;
        std::thread accept_thread([this, recv_port, &recv_sock]() {
              TCPSocket listen_sock;
              listen_sock.Create();
              listen_sock.SetKeepAlive(true);
              listen_sock.Bind(SockAddr("0.0.0.0", recv_port));
              listen_sock.Listen();
              SockAddr tmp_addr;
              recv_sock = listen_sock.Accept(&tmp_addr);
              LOG(INFO) << "[Host] Recv from " << tmp_addr.AsString();
              listen_sock.Close();
        });

        TCPSocket send_sock;
        send_sock.Create();
        send_sock.SetKeepAlive(true);
        bool ok = false;
        for (int i = 0; i < 100; ++i) {
           if (send_sock.Connect(SockAddr(node_hosts_[next_node_id].c_str(), send_port))) { ok = true; break; }
           std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        ICHECK(ok);
        accept_thread.join();


        tcp_in_ch_ = std::make_unique<DiscoRingChannel>(recv_sock.sockfd);
        recv_sock.sockfd = Socket::INVALID_SOCKET;
        tcp_out_ch_ = std::make_unique<DiscoRingChannel>(send_sock.sockfd);
        send_sock.sockfd = Socket::INVALID_SOCKET;


        // 0:read 1:write
        int sess_fds[2];
        ICHECK_EQ(::pipe(sess_fds), 0);

        proxy_out_to_tcp_= local_session_->RerouteRingIn(std::make_unique<DiscoRingChannel>(sess_fds[0]));
        ICHECK(proxy_out_to_tcp_);

        proxy_in_from_tcp_= std::make_unique<DiscoRingChannel>(sess_fds[1]);
       

        send_thread_ = std::thread(ProxyLoop, proxy_out_to_tcp_.get(), tcp_out_ch_.get(), "controller-send");
        recv_thread_ = std::thread(ProxyLoop, tcp_in_ch_.get(), proxy_in_from_tcp_.get(), "controller-recv");
        const int num_workers_ = num_nodes_ * num_workers_per_node_;
        for (int worker_id = 1; worker_id < num_workers_; ++worker_id) { this->SyncWorker(worker_id); }
        
        LOG(INFO) << "[Host] Controller: All workers are ready. Initialization complete.";


    }// build ring

  } //explictit

  int64_t GetNumWorkers() final { return num_nodes_ * num_workers_per_node_; }

  ffi::Any DebugGetFromRemote(int64_t reg_id, int worker_id) final {
    int node_id = worker_id / num_workers_per_node_;
    if (node_id == 0) {
      return local_session_->DebugGetFromRemote(reg_id, worker_id);
    } else {
      AnyView packed_args[5];
      ffi::PackedArgs::Fill(packed_args, static_cast<int>(DiscoSocketAction::kSend), worker_id,
                            static_cast<int>(DiscoAction::kDebugGetFromRemote), reg_id, worker_id);
      remote_channels_[node_id - 1]->Send(ffi::PackedArgs(packed_args, 5));
      ffi::PackedArgs args = this->RecvReplyPacked(worker_id);
      ICHECK_EQ(args.size(), 2);
      ICHECK(static_cast<DiscoAction>(args[0].cast<int>()) == DiscoAction::kDebugGetFromRemote);
      ffi::Any result;
      result = args[1];
      return result;
    }
  }

  void DebugSetRegister(int64_t reg_id, AnyView value, int worker_id) final {
    int node_id = worker_id / num_workers_per_node_;
    if (node_id == 0) {
      local_session_->DebugSetRegister(reg_id, value, worker_id);
    } else {
      ObjectRef wrapped{nullptr};
      if (auto opt_obj = value.as<ObjectRef>()) {
        wrapped = DiscoDebugObject::Wrap(value);
        value = wrapped;
      }
      {
        AnyView packed_args[6];
        ffi::PackedArgs::Fill(packed_args, static_cast<int>(DiscoSocketAction::kSend), worker_id,
                              static_cast<int>(DiscoAction::kDebugSetRegister), reg_id, worker_id,
                              value);
        remote_channels_[node_id - 1]->Send(ffi::PackedArgs(packed_args, 6));
      }
      ffi::Any result;
      ffi::PackedArgs args = this->RecvReplyPacked(worker_id);
      ICHECK_EQ(args.size(), 1);
      ICHECK(static_cast<DiscoAction>(args[0].cast<int>()) == DiscoAction::kDebugSetRegister);
    }
  }

  void BroadcastPacked(const ffi::PackedArgs& args) final {
    local_session_->BroadcastPacked(args);
    std::vector<AnyView> packed_args(args.size() + 2);
    ffi::PackedArgs::Fill(packed_args.data(), static_cast<int>(DiscoSocketAction::kSend), -1);
    std::copy(args.data(), args.data() + args.size(), packed_args.begin() + 2);
    for (auto& channel : remote_channels_) {
      channel->Send(ffi::PackedArgs(packed_args.data(), packed_args.size()));
    }
  }

  void SendPacked(int worker_id, const ffi::PackedArgs& args) final {
    int node_id = worker_id / num_workers_per_node_;
    if (node_id == 0) {
      local_session_->SendPacked(worker_id, args);
      return;
    }
    std::vector<AnyView> packed_args(args.size() + 2);
    ffi::PackedArgs::Fill(packed_args.data(), static_cast<int>(DiscoSocketAction::kSend),
                          worker_id);
    std::copy(args.data(), args.data() + args.size(), packed_args.begin() + 2);
    remote_channels_[node_id - 1]->Send(ffi::PackedArgs(packed_args.data(), packed_args.size()));
  }

  ffi::PackedArgs RecvReplyPacked(int worker_id) final {
    int node_id = worker_id / num_workers_per_node_;
    if (node_id == 0) {
      return local_session_->RecvReplyPacked(worker_id);
    }
    AnyView packed_args[2];
    ffi::PackedArgs::Fill(packed_args, static_cast<int>(DiscoSocketAction::kReceive), worker_id);
    remote_channels_[node_id - 1]->Send(ffi::PackedArgs(packed_args, 2));
    return remote_channels_[node_id - 1]->Recv();
  }

  void AppendHostTensor(const Tensor& host_array) final {
    local_session_->AppendHostTensor(host_array);
  }

  void Shutdown() final {

    local_session_->Shutdown();
    if (proxy_out_to_tcp_)  proxy_out_to_tcp_->Close();
    if (proxy_in_from_tcp_) proxy_in_from_tcp_->Close();
    if (tcp_in_ch_)        tcp_in_ch_->Close();
    if (tcp_out_ch_)       tcp_out_ch_->Close();

    std::vector<AnyView> packed_args(2);
    ffi::PackedArgs::Fill(packed_args.data(), static_cast<int>(DiscoSocketAction::kShutdown), -1);
    for (auto& channel : remote_channels_) {
      channel->Send(ffi::PackedArgs(packed_args.data(), packed_args.size()));
    }
    for (auto& socket : remote_sockets_) {
      socket.Close();
    }
    remote_sockets_.clear();
    remote_channels_.clear();
    if (!socket_.IsClosed()) {
      socket_.Close();
    }
    Socket::Finalize();

    if (send_thread_.joinable()) send_thread_.join();
    if (recv_thread_.joinable()) recv_thread_.join();
  }
  

  ~SocketSessionObj() { Shutdown(); }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("runtime.disco.SocketSession", SocketSessionObj,
                                    BcastSessionObj);
  int num_nodes_;
  int num_workers_per_node_;
  TCPSocket socket_;
  std::vector<TCPSocket> remote_sockets_;
  std::vector<std::unique_ptr<DiscoSocketChannel>> remote_channels_;
  std::vector<ffi::String> node_hosts_;
  BcastSession local_session_{nullptr};

  private:
   std::unique_ptr<DiscoRingChannel> proxy_out_to_tcp_;
   std::unique_ptr<DiscoRingChannel> proxy_in_from_tcp_;
   std::unique_ptr<DiscoRingChannel> tcp_in_ch_;
   std::unique_ptr<DiscoRingChannel> tcp_out_ch_;
   std::thread send_thread_;
   std::thread recv_thread_;
};

class RemoteSocketSession {
 public:
  explicit RemoteSocketSession(const ffi::String& server_host, int server_port,
                               int num_local_workers) {
    socket_.Create();
    socket_.SetKeepAlive(true);
    SockAddr server_addr{server_host.c_str(), server_port};
    Socket::Startup();
    if (!socket_.Connect(server_addr)) {
      LOG(FATAL) << "Failed to connect to server " << server_addr.AsString()
                 << ", errno = " << Socket::GetLastErrorCode();
    }
    channel_ = std::make_unique<DiscoSocketChannel>(socket_);
    ffi::PackedArgs metadata = channel_->Recv();
    ICHECK_EQ(metadata.size(), 5);
    num_nodes_ = metadata[0].cast<int>();
    num_workers_per_node_ = metadata[1].cast<int>();
    num_groups_ = metadata[2].cast<int>();
    node_id_ = metadata[3].cast<int>();
    build_ring = metadata[4].cast<bool>();
    CHECK_GE(num_local_workers, num_workers_per_node_);
    InitLocalSession();
  }

  void MainLoop() {
    while (true) {
      ffi::PackedArgs args = channel_->Recv();
      DiscoSocketAction action = static_cast<DiscoSocketAction>(args[0].cast<int>());
      int worker_id = args[1].cast<int>();
      int local_worker_id = worker_id - node_id_ * num_workers_per_node_;
      switch (action) {
        case DiscoSocketAction::kSend: {
          args = args.Slice(2);
          if (worker_id == -1) {
            local_session_->BroadcastPacked(args);
          } else {
            local_session_->SendPacked(local_worker_id, args);
          }
          break;
        }
        case DiscoSocketAction::kReceive: {
          args = local_session_->RecvReplyPacked(local_worker_id);
          channel_->Reply(args);
          break;
        }
        case DiscoSocketAction::kShutdown: {
          local_session_->Shutdown();
          LOG(INFO) << "Connection closed by remote controller.";
          return;
        }
        default:
          LOG(FATAL) << "Invalid action " << static_cast<int>(action);
      }
    }
  }

  ~RemoteSocketSession() {  
    
    if (local_session_.defined()) local_session_->Shutdown();
    if (proxy_out_to_tcp_)        proxy_out_to_tcp_->Close();
    if (proxy_in_from_tcp_)       proxy_in_from_tcp_->Close();
    if (tcp_in_ch_)               tcp_in_ch_->Close();
    if (tcp_out_ch_)              tcp_out_ch_->Close();
    if (send_thread_.joinable()) send_thread_.join();
    if (recv_thread_.joinable()) recv_thread_.join();

    if (!socket_.IsClosed()) socket_.Close();
    Socket::Finalize();

  }

 private:
  void InitLocalSession() {
    const auto f_create_local_session =
        tvm::ffi::Function::GetGlobal("runtime.disco.create_socket_session_local_workers");
    local_session_ = ((*f_create_local_session)(num_workers_per_node_, build_ring)).cast<BcastSession>();

    DRef f_init_workers =
        local_session_->GetGlobalFunc("runtime.disco.socket_session_init_workers");
    local_session_->CallPacked(f_init_workers, num_nodes_, node_id_, num_groups_,
                               num_workers_per_node_);

    if( build_ring && num_nodes_ > 1)  {
      
      ffi::PackedArgs host_info = channel_->Recv();
      const int nNodes = host_info.size() - 1;
      const int base_ring_port = host_info[0].cast<int>();
      std::vector<ffi::String> node_hosts;
      node_hosts.reserve(nNodes);
      for (int i = 0; i < nNodes; ++i) node_hosts.push_back(host_info[1 + i].cast<ffi::String>());


      const int next_node_id   = (node_id_ + 1) % nNodes;
      const int recv_port      = base_ring_port + node_id_;
      const int send_port      = base_ring_port + next_node_id;

      LOG(INFO) << "[Remote] remote node_id=" << node_id_ << " listen=:" << recv_port
            << " connect_to=" << node_hosts[next_node_id].c_str() << ":" << send_port;


      TCPSocket recv_sock;
      std::thread accept_thread([this, recv_port, &recv_sock]() {
              TCPSocket listen_sock;
              listen_sock.Create();
              listen_sock.SetKeepAlive(true);
              listen_sock.Bind(SockAddr("0.0.0.0", recv_port));
              listen_sock.Listen();
              SockAddr tmp_addr;
              recv_sock = listen_sock.Accept(&tmp_addr);
              LOG(INFO) << "[Remote] Recv from " << tmp_addr.AsString();
              listen_sock.Close();
       });

      TCPSocket send_sock;
      send_sock.Create();
      send_sock.SetKeepAlive(true);
      bool ok = false;
      for (int i = 0; i < 100; ++i) {
           if (send_sock.Connect(SockAddr(node_hosts[next_node_id].c_str(), send_port))) { ok = true; break; }
           std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
      ICHECK(ok);
      LOG(INFO) << "[Remote] node_id=" << node_id_ << " outbound connected to " << node_hosts[next_node_id].c_str() << ":" << send_port;
      accept_thread.join();
      LOG(INFO) << "[Remote] node_id=" << node_id_ << " ring TCP ready";


      tcp_in_ch_  = std::make_unique<DiscoRingChannel>(recv_sock.sockfd);
      recv_sock.sockfd = Socket::INVALID_SOCKET;
      tcp_out_ch_ = std::make_unique<DiscoRingChannel>(send_sock.sockfd);
      send_sock.sockfd = Socket::INVALID_SOCKET;

  
      // 0:read 1:write
      int sess_fds[2];
      ICHECK_EQ(::pipe(sess_fds), 0);

      proxy_out_to_tcp_= local_session_->RerouteRingIn(std::make_unique<DiscoRingChannel>(sess_fds[0]));
      ICHECK(proxy_out_to_tcp_);

      proxy_in_from_tcp_= std::make_unique<DiscoRingChannel>(sess_fds[1]);

 

  
      const std::string tag_prefix = "node_" + std::to_string(node_id_);
      send_thread_ = std::thread(ProxyLoop, proxy_out_to_tcp_.get(), tcp_out_ch_.get(),tag_prefix + "-send");
      recv_thread_ = std::thread(ProxyLoop, tcp_in_ch_.get(), proxy_in_from_tcp_.get(),tag_prefix + "-recv");
      LOG(INFO) << "[Proxy] " << tag_prefix << ": send/recv threads started";
        
    }//build ring
  }

  TCPSocket socket_;
  BcastSession local_session_{nullptr};
  std::unique_ptr<DiscoSocketChannel> channel_;
  int num_nodes_{-1};
  int node_id_{-1};
  int num_groups_{-1};
  int num_workers_per_node_{-1};
  bool build_ring = false;


  std::unique_ptr<DiscoRingChannel> proxy_out_to_tcp_;
  std::unique_ptr<DiscoRingChannel> proxy_in_from_tcp_;
  std::unique_ptr<DiscoRingChannel> tcp_in_ch_;
  std::unique_ptr<DiscoRingChannel> tcp_out_ch_;
  std::thread send_thread_;
  std::thread recv_thread_;
};

void RemoteSocketSessionEntryPoint(const ffi::String& server_host, int server_port,
                                   int num_local_workers) {
  RemoteSocketSession proxy(server_host, server_port, num_local_workers);
  proxy.MainLoop();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("runtime.disco.RemoteSocketSession", RemoteSocketSessionEntryPoint);
}

Session SocketSession(int num_nodes, int num_workers_per_node, int num_groups,
                      const ffi::String& host, int port,
                      bool build_ring) {
  auto n =
      ffi::make_object<SocketSessionObj>(num_nodes, num_workers_per_node, num_groups, host, port, build_ring);
  return Session(n);
}




TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<SocketSessionObj>();
  refl::GlobalDef()
      .def("runtime.disco.SocketSession", SocketSession)
      .def("runtime.disco.socket_session_init_workers",
           [](int num_nodes, int node_id, int num_groups, int num_workers_per_node) {
             DiscoWorker* worker = DiscoWorker::ThreadLocal();
             worker->num_groups = num_groups;
             worker->worker_id = worker->worker_id + node_id * num_workers_per_node;
             worker->num_workers = num_nodes * num_workers_per_node;
             LOG(INFO) << "[Disco_worker_" << worker->worker_id << "]" << " Initializing worker group with " << num_nodes << " nodes, "
                       << num_workers_per_node << " workers per node, and " << num_groups
                       << " groups.";
           });
  }

}  // namespace runtime
}  // namespace tvm
