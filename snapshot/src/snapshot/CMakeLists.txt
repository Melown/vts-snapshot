define_module(BINARY vts-snapshot
  DEPENDS vtsoffscreen>=1.0 geo>=1.28 service>=1.6
  geometry>=1.10 imgproc>=1.23
  glsupport>=1.0 dbglog>=1.4
  )

set(vts-snapshot_SOURCES
  main.cpp
  )

add_executable(vts-snapshot ${vts-snapshot_SOURCES})
target_link_libraries(vts-snapshot ${MODULE_LIBRARIES})
buildsys_target_compile_definitions(vts-snapshot ${MODULE_DEFINITIONS})
buildsys_binary(vts-snapshot)
