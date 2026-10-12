#pragma once

// Live SocketCAN (CAN-FD) reader for the Columbus FC 4D radar. Owns a dedicated background thread
// because AF_CAN recv() blocks indefinitely (radar powered off / cable unplugged must never stall
// the Qt event loop) -- this is the app's first background thread. Every other tab reads its sensor
// (camera) synchronously on the Qt main thread via a QTimer tick, which is fine for a non-blocking
// V4L2 read but wrong for a socket read that can block forever.
//
// Usage: construct, call start(interface_name) once, poll snapshot() from the Qt main thread's
// QTimer tick (cheap, mutex-guarded copy), call stop() (or let the destructor do it) to join the
// thread. Implementation lives in socketcan_reader.cpp since it needs real socket syscalls.

#include "bev/radarcam/can_decoder.hpp"
#include "bev/radarcam/drwig_decoder.hpp"

#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace bev {
namespace radarcam {

// Which radar's CAN protocol the reader decodes.
enum class RadarProtocol {
    kColumbusFc4d,  // CAN-FD raw detections, IDs 0x281-0x284 (can_decoder.hpp)
    kDrwigLrrPro,   // classic CAN 2D tracked objects, one radarId's ID range (drwig_decoder.hpp)
};

struct DwellSnapshot {
    std::vector<RadarDetection> detections;
    int frame_id = 0;
    int radar_id = 0;
    long total_messages = 0;
};

class SocketCanReader {
public:
    SocketCanReader();
    ~SocketCanReader();

    SocketCanReader(const SocketCanReader&) = delete;
    SocketCanReader& operator=(const SocketCanReader&) = delete;

    // Opens `interface_name` (e.g. "can1", "vcan0"), enables CAN-FD frame reception, and starts the
    // background reader thread. Returns false (with `error` populated) on any socket/bind failure.
    // `drwig_radar_id` (0..5) selects which radar on a shared bus to decode; only used for
    // RadarProtocol::kDrwigLrrPro.
    bool start(const std::string& interface_name, std::string& error,
        RadarProtocol protocol = RadarProtocol::kColumbusFc4d, int drwig_radar_id = 0);

    // Stops the background thread and closes the socket. Safe to call multiple times; the
    // destructor also calls this.
    void stop();

    bool is_running() const { return running_.load(); }

    // Cheap, mutex-guarded copy of the latest published dwell. Safe to call from the Qt main thread.
    DwellSnapshot snapshot() const;

private:
    void reader_loop(int fd);

    std::thread thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> stop_requested_{false};

    mutable std::mutex mutex_;
    DwellSnapshot snapshot_;
    RadarProtocol protocol_ = RadarProtocol::kColumbusFc4d;  // set in start(), before the thread runs
    RadarDwellAccumulator accumulator_;  // only touched by reader_loop -- no lock needed there
    DrwigLrrProDecoder drwig_decoder_;   // likewise
    long total_messages_ = 0;
};

}  // namespace radarcam
}  // namespace bev
