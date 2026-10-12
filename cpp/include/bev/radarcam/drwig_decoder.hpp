#pragma once

// DRWIG LRR Pro FRS radar CAN decode (classic CAN, 8-byte frames), ported from the drwig_lrr_pro
// repo (cpp/include/drwig_lrr_pro/{signal_codec,checksum,radar_id,can_types}.hpp + 220pro.dbc).
// Only the pieces the calibration needs: the 40 tracked-object slots (Part1 = valid flag, Part2 =
// position/velocity) and the FRS_Status frame, used here as the once-per-cycle "dwell complete"
// marker. Re-sync by hand against that repo if its decode ever changes -- no shared package links
// the two.
//
// Unlike the Columbus FC 4D (can_decoder.hpp), this radar only reports 2D TRACKED OBJECTS: x/y
// position + velocity, no elevation, no RCS, no raw per-frame detections. Every RadarDetection
// produced here therefore has z = 0, elevation = 0 and rcs_dbsm = 0 -- the calibration handles
// that via ClutterFilterParams::planar/has_rcs and the planar solve (planar_solve.hpp).
//
// Radar frame: x forward, y POSITIVE TO THE RADAR'S RIGHT (220pro.dbc's raw sign, kept as-is --
// NOT the y-left convention geometry.hpp documents for the Columbus radar), z down, so the axes
// stay right-handed. The coarse extrinsic in the radar-cam config must be written for this frame.
//
// Several radars can share one bus: each radar's configured radarId (0..5) adds a fixed offset to
// every CAN ID it transmits (see radar_id_can_offset()), so one decoder instance decodes exactly
// one radar and ignores the rest.

#include "bev/radarcam/can_decoder.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace bev {
namespace radarcam {

namespace drwig {

inline constexpr int kNumSlots = 40;
inline constexpr uint32_t kPart2BaseId = 0x20;   // FRS_Obj_<slot>_Part2of2: position/velocity
inline constexpr uint32_t kPart1BaseId = 0x50;   // FRS_Obj_<slot>_Part1of2: id/valid/existence
inline constexpr uint32_t kStatusId = 0x80;      // FRS_Status, once per radar cycle
inline constexpr size_t kChecksumByteIndex = 7;  // every FRS_* frame: CRC over bytes 0-6, stored in byte 7

// Vendor "Radar ID Configuration Instructions" (drwig_lrr_pro radar_id.hpp): radarId -> CAN-ID offset.
// 0 Front-Center, 1 Front-Left, 2 Front-Right, 3 Rear-Center, 4 Rear-Left, 5 Rear-Right.
inline constexpr uint32_t radar_id_can_offset(int radar_id) {
    switch (radar_id) {
        case 1: return 0x130;
        case 2: return 0x220;
        case 3: return 0x340;
        case 4: return 0x530;
        case 5: return 0x710;
        default: return 0x000;
    }
}

// CRC-8 SAE J1850 / AUTOSAR CRC8H2F: poly 0x2F, init 0xFF, xorout 0xFF, no reflection
// (vendor-confirmed FRS_* checksum, drwig_lrr_pro checksum.hpp).
inline uint8_t crc8_sae_j1850_h2f(const uint8_t* data, size_t len) {
    uint8_t crc = 0xFF;
    for (size_t i = 0; i < len; ++i) {
        crc ^= data[i];
        for (int bit = 0; bit < 8; ++bit) {
            crc = (crc & 0x80) ? static_cast<uint8_t>((crc << 1) ^ 0x2F) : static_cast<uint8_t>(crc << 1);
        }
    }
    return static_cast<uint8_t>(crc ^ 0xFF);
}

// DBC Motorola (@0) extraction: start_bit labels the signal's MSB (bit 8b+7..8b within byte b); the
// signal walks down toward its LSB, wrapping to bit 7 of the next byte (drwig_lrr_pro signal_codec.hpp).
inline uint32_t extract_motorola(const uint8_t* data, int start_bit, int length) {
    uint32_t result = 0;
    int byte_pos = start_bit / 8;
    int bit_pos = start_bit % 8;
    for (int i = 0; i < length; ++i) {
        result = (result << 1) | ((data[byte_pos] >> bit_pos) & 1u);
        if (bit_pos == 0) {
            bit_pos = 7;
            ++byte_pos;
        } else {
            --bit_pos;
        }
    }
    return result;
}

}  // namespace drwig

// One radar's slot table. Feed it every frame from the bus via handle_frame(); a completed cycle
// (this radar's FRS_Status frame) bumps frame_id() and refreshes detections() with every slot that
// is currently valid and whose both halves were updated within the stale timeout.
class DrwigLrrProDecoder {
public:
    // Matches drwig_lrr_pro parser.cpp's kStaleTimeoutMs: 4 x the radar's 50ms cycle.
    static constexpr int64_t kStaleTimeoutMs = 200;

    explicit DrwigLrrProDecoder(int radar_id = 0) : offset_(drwig::radar_id_can_offset(radar_id)) {}

    // Returns true when this frame completed a radar cycle (i.e. detections() was just refreshed).
    // Frames from other radars, unknown IDs, short payloads and bad checksums are ignored (the latter
    // counted in bad_checksum()).
    bool handle_frame(uint32_t can_id, const uint8_t* data, size_t len, int64_t now_ms) {
        const bool is_status = can_id == drwig::kStatusId + offset_;
        const bool is_part1 = can_id >= drwig::kPart1BaseId + offset_ &&
                              can_id < drwig::kPart1BaseId + offset_ + drwig::kNumSlots;
        const bool is_part2 = can_id >= drwig::kPart2BaseId + offset_ &&
                              can_id < drwig::kPart2BaseId + offset_ + drwig::kNumSlots;
        if (!is_status && !is_part1 && !is_part2) return false;
        if (len < 8) return false;
        if (drwig::crc8_sae_j1850_h2f(data, drwig::kChecksumByteIndex) != data[drwig::kChecksumByteIndex]) {
            ++bad_checksum_;
            return false;
        }

        if (is_part1) {
            Slot& s = slots_[can_id - drwig::kPart1BaseId - offset_];
            s.valid = drwig::extract_motorola(data, 23, 1) != 0;              // Obj_ValidFlag
            s.motion_pattern = static_cast<int>(drwig::extract_motorola(data, 31, 3));  // Obj_MotionPattern
            s.part1_ms = now_ms;
            s.part1_seen = true;
            return false;
        }
        if (is_part2) {
            Slot& s = slots_[can_id - drwig::kPart2BaseId - offset_];
            s.x_m = drwig::extract_motorola(data, 31, 14) * 0.015625;                // Obj_XPos
            s.y_m = drwig::extract_motorola(data, 12, 13) * 0.015625 - 64.0;         // Obj_YPos
            s.vx_mps = drwig::extract_motorola(data, 7, 11) * 0.1 - 102.4;           // Obj_XVelRel
            s.vy_mps = drwig::extract_motorola(data, 47, 11) * 0.1 - 102.4;          // Obj_YVelRel
            s.part2_ms = now_ms;
            s.part2_seen = true;
            return false;
        }

        // FRS_Status: end of one radar cycle -- publish the current fresh, valid slots.
        detections_.clear();
        for (const auto& s : slots_) {
            if (!s.valid || !s.part1_seen || !s.part2_seen) continue;
            if (now_ms - s.part1_ms >= kStaleTimeoutMs || now_ms - s.part2_ms >= kStaleTimeoutMs) continue;
            detections_.push_back(to_detection(s));
        }
        ++frame_id_;
        return true;
    }

    const std::vector<RadarDetection>& detections() const { return detections_; }
    int frame_id() const { return frame_id_; }
    long bad_checksum() const { return bad_checksum_; }

private:
    struct Slot {
        bool valid = false;
        int motion_pattern = 0;  // 0 unknown, 2 stopped, 3 moving (per vendor protocol doc)
        double x_m = 0.0, y_m = 0.0, vx_mps = 0.0, vy_mps = 0.0;
        bool part1_seen = false, part2_seen = false;
        int64_t part1_ms = 0, part2_ms = 0;
    };

    static RadarDetection to_detection(const Slot& s) {
        RadarDetection d;
        d.range_m = std::hypot(s.x_m, s.y_m);
        d.azimuth_rad = std::atan2(s.y_m, s.x_m);
        d.elevation_rad = 0.0;
        // Radial (range-rate) velocity -- what the clutter filter's Doppler gate expects.
        d.vel_mps = d.range_m > 1e-6 ? (s.x_m * s.vx_mps + s.y_m * s.vy_mps) / d.range_m : 0.0;
        d.rcs_dbsm = 0.0;  // not reported by this radar
        d.snr_db = 0.0;
        d.is_stationary = s.motion_pattern != 3;
        // Built through spherical_to_cartesian so a capture saved as (range, azimuth, elevation)
        // round-trips to exactly this position on load (capture_io.hpp).
        d.position = spherical_to_cartesian(d.range_m, d.azimuth_rad, d.elevation_rad);
        return d;
    }

    uint32_t offset_;
    std::array<Slot, drwig::kNumSlots> slots_{};
    std::vector<RadarDetection> detections_;
    int frame_id_ = 0;
    long bad_checksum_ = 0;
};

}  // namespace radarcam
}  // namespace bev
