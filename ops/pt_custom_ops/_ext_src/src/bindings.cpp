#include "group_points.h"
#include "interpolate.h"
#include "masked_ordered_ball_query.h"
#include "masked_grid_subsampling.h"
#include "masked_nearest_query.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

  m.def("group_points", &group_points);
  m.def("group_points_grad", &group_points_grad);

  m.def("three_nn", &three_nn);
  m.def("three_interpolate", &three_interpolate);
  m.def("three_interpolate_grad", &three_interpolate_grad);

  m.def("masked_ordered_ball_query", &masked_ordered_ball_query);
  m.def("masked_grid_subsampling", &masked_grid_subsampling);

  m.def("masked_nearest_query", &masked_nearest_query);
}
