find_package(Catch2 3 REQUIRED)
include(Catch)

add_executable(radarcam_tests
    test_trivial.cpp
    test_config.cpp
    test_geometry.cpp
    test_can_decoder.cpp
    test_socketcan_vcan.cpp
    test_capture_io.cpp
    test_board_pnp.cpp
    test_clutter_filter.cpp
    test_kabsch_ransac.cpp
    test_gate_iterate.cpp
    test_refine_ceres.cpp
    test_validate.cpp
    test_diagnostics_render.cpp
    test_drwig_planar.cpp
)
target_link_libraries(radarcam_tests PRIVATE bev_core bev_radarcam Catch2::Catch2WithMain)
target_compile_definitions(radarcam_tests PRIVATE FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures")
set(BEV_RADARCAM_PREVIEW_DIR "$ENV{HOME}/.cache/openadas_radarcam_previews" CACHE PATH
    "Scratch directory Phase 11's diagnostics-render test writes preview PNGs to for manual visual review")
file(MAKE_DIRECTORY "${BEV_RADARCAM_PREVIEW_DIR}")
target_compile_definitions(radarcam_tests PRIVATE PREVIEW_DIR="${BEV_RADARCAM_PREVIEW_DIR}")

catch_discover_tests(radarcam_tests PROPERTIES SKIP_RETURN_CODE 4)
