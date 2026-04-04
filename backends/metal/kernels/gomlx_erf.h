#pragma once

#include <metal_stdlib>

using namespace metal;

// MSL does not provide erf(); Abramowitz & Stegun formula 7.1.26 (~1.5e-7 max error on float).
METAL_FUNC float gomlx_erf(float x)
{
    const float a1 = 0.254829592f;
    const float a2 = -0.284496736f;
    const float a3 = 1.421413741f;
    const float a4 = -1.453152027f;
    const float a5 = 1.061405429f;
    const float p = 0.3275911f;

    const float sx = copysign(1.0f, x);
    x = fabs(x);
    const float t = 1.0f / (1.0f + p * x);
    const float y =
        1.0f -
        (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x);
    return sx * y;
}
