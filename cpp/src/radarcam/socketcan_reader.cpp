#include "bev/radarcam/socketcan_reader.hpp"

#include <linux/can.h>
#include <linux/can/raw.h>
#include <net/if.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstring>

namespace bev {
namespace radarcam {

SocketCanReader::SocketCanReader() = default;

SocketCanReader::~SocketCanReader() { stop(); }

bool SocketCanReader::start(
    const std::string& interface_name, std::string& error, RadarProtocol protocol, int drwig_radar_id) {
    if (running_.load()) {
        error = "SocketCanReader is already running.";
        return false;
    }
    // Fresh decode state per run, so switching protocol/radarId between runs can't mix old slots in.
    protocol_ = protocol;
    accumulator_ = RadarDwellAccumulator{};
    drwig_decoder_ = DrwigLrrProDecoder(drwig_radar_id);
    total_messages_ = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        snapshot_ = DwellSnapshot{};
    }

    const int fd = socket(AF_CAN, SOCK_RAW, CAN_RAW);
    if (fd < 0) {
        error = std::string("socket(AF_CAN) failed: ") + std::strerror(errno);
        return false;
    }

    constexpr int kEnableFd = 1;
    if (setsockopt(fd, SOL_CAN_RAW, CAN_RAW_FD_FRAMES, &kEnableFd, sizeof(kEnableFd)) < 0) {
        error = std::string("setsockopt(CAN_RAW_FD_FRAMES) failed: ") + std::strerror(errno);
        ::close(fd);
        return false;
    }

    struct ifreq ifr {};
    std::strncpy(ifr.ifr_name, interface_name.c_str(), IFNAMSIZ - 1);
    if (ioctl(fd, SIOCGIFINDEX, &ifr) < 0) {
        error = "Interface '" + interface_name + "' not found: " + std::strerror(errno);
        ::close(fd);
        return false;
    }

    struct sockaddr_can addr {};
    addr.can_family = AF_CAN;
    addr.can_ifindex = ifr.ifr_ifindex;
    if (bind(fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
        error = std::string("bind() failed: ") + std::strerror(errno);
        ::close(fd);
        return false;
    }

    // Recv timeout so the background thread wakes periodically to check stop_requested_ instead of
    // blocking in recv() forever -- the simplest clean way to join a thread parked on a blocking
    // socket read without a self-pipe/select setup.
    struct timeval tv {};
    tv.tv_sec = 0;
    tv.tv_usec = 200000;  // 200ms
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    stop_requested_.store(false);
    running_.store(true);
    thread_ = std::thread(&SocketCanReader::reader_loop, this, fd);
    return true;
}

void SocketCanReader::stop() {
    if (!running_.load() && !thread_.joinable()) return;
    stop_requested_.store(true);
    if (thread_.joinable()) thread_.join();
    running_.store(false);
}

void SocketCanReader::reader_loop(int fd) {
    struct canfd_frame frame {};
    while (!stop_requested_.load()) {
        const ssize_t n = read(fd, &frame, sizeof(frame));
        if (n < 0) {
            continue;  // recv timeout (EAGAIN/EWOULDBLOCK) or interrupted -- just re-check the flag
        }
        if (n != static_cast<ssize_t>(CANFD_MTU) && n != static_cast<ssize_t>(CAN_MTU)) {
            continue;  // malformed/short read
        }

        ++total_messages_;
        const canid_t cid = frame.can_id & CAN_EFF_MASK;

        if (protocol_ == RadarProtocol::kDrwigLrrPro) {
            const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            const bool cycle_done = drwig_decoder_.handle_frame(cid, frame.data, frame.len, now_ms);

            std::lock_guard<std::mutex> lock(mutex_);
            if (cycle_done) {
                snapshot_.detections = drwig_decoder_.detections();
                snapshot_.frame_id = drwig_decoder_.frame_id();
            }
            snapshot_.total_messages = total_messages_;
            continue;
        }

        const std::vector<uint8_t> data(frame.data, frame.data + frame.len);

        if (cid == 0x281) {
            accumulator_.on_header(RadarCanFdParser::parse_rdi_header(data));
        } else if (cid == 0x282 || cid == 0x283 || cid == 0x284) {
            accumulator_.on_body(data);

            std::lock_guard<std::mutex> lock(mutex_);
            snapshot_.detections = accumulator_.detections();
            snapshot_.frame_id = accumulator_.frame_id();
            snapshot_.radar_id = accumulator_.radar_id();
            snapshot_.total_messages = total_messages_;
        }
    }
    ::close(fd);
}

DwellSnapshot SocketCanReader::snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return snapshot_;
}

}  // namespace radarcam
}  // namespace bev
