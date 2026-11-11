#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

// D16 complex value (group Z_8 |>< Z_2 of order 16):
// Coordinates (x, y) live exclusively in the closed 22.5-degree fundamental wedge:
//   0 <= y <= (sqrt(2) - 1) * x.
// The orientation is a 16-bit sector mask (std::uint16_t):
//   - Interior of sector s in 0..15 : one bit (1u << s)
//   - Boundary (multiples of 22.5°) : two adjacent bits
//   - Zero (State::Indifferent)     : 0xFFFF (all 16 orbit positions active)
//   - NaN  (State::Undefined)       : 0x0000
//
// All 16 dihedral operations of Z_8 |>< Z_2 act PURELY on the 16-bit mask
// without modifying (x, y).

struct D16Complex final
{
    static constexpr double SQRT2 = 1.41421356237309504880;
    static constexpr double INV_SQRT2 = 0.70710678118654752440;
    static constexpr double TAN_PI_8 = 0.41421356237309504880; // sqrt(2) - 1
    static constexpr double EPS_BOUND = 1e-12;

    double x = 0.0;
    double y = 0.0;
    std::uint16_t rotation = 0xFFFF;

    static constexpr std::uint16_t rol(std::uint16_t m, unsigned n = 1) noexcept
    {
        n &= 15u;
        return static_cast<std::uint16_t>((static_cast<unsigned>(m) << n) |
                                          (static_cast<unsigned>(m) >> ((16u - n) & 15u)));
    }
    static constexpr std::uint16_t ror(std::uint16_t m, unsigned n = 1) noexcept
    {
        return rol(m, (16u - (n & 15u)) & 15u);
    }

    static constexpr std::uint16_t reverse_bits(std::uint16_t m) noexcept
    {
        m = static_cast<std::uint16_t>(((m & 0xFF00u) >> 8) | ((m & 0x00FFu) << 8));
        m = static_cast<std::uint16_t>(((m & 0xF0F0u) >> 4) | ((m & 0x0F0Fu) << 4));
        m = static_cast<std::uint16_t>(((m & 0xCCCCu) >> 2) | ((m & 0x3333u) << 2));
        m = static_cast<std::uint16_t>(((m & 0xAAAAu) >> 1) | ((m & 0x5555u) << 1));
        return m;
    }

    static constexpr D16Complex zero() noexcept { return {0.0, 0.0, 0xFFFF}; }
    static constexpr D16Complex nan() noexcept { return {0.0, 0.0, 0x0000}; }

    constexpr bool is_zero() const noexcept { return rotation == 0xFFFF; }
    constexpr bool is_nan() const noexcept { return rotation == 0x0000; }

    static constexpr int bit_index(std::uint16_t m) noexcept
    {
        int k = 0;
        while ((m >>= 1u) != 0u)
            ++k;
        return k;
    }

    static constexpr bool is_one_hot(std::uint16_t m) noexcept
    {
        return m != 0u && (m & static_cast<std::uint16_t>(m - 1u)) == 0u;
    }

    static constexpr int even_anchor(std::uint16_t m) noexcept
    {
        if (is_one_hot(m))
            return bit_index(m);
        for (int k = 0; k < 16; k += 2)
        {
            if (m & static_cast<std::uint16_t>(1u << k))
                return k;
        }
        return -1;
    }

    // Returns canonical state in 0..15, 16 for Indifferent (0xFFFF), 17 for Undefined (0x0000)
    constexpr int state() const noexcept
    {
        if (rotation == 0xFFFF)
            return 16;
        if (rotation == 0x0000)
            return 17;
        const std::uint16_t t = static_cast<std::uint16_t>(rotation & rol(rotation, 1));
        return bit_index(t != 0u ? t : rotation);
    }

    // C++17 constexpr helpers; do not depend on constexpr <cmath> support.
    static constexpr double abs_value(double value) noexcept
    {
        return value < 0.0 ? -value : value;
    }
    static constexpr bool finite_value(double value) noexcept
    {
        return value >= -std::numeric_limits<double>::max() &&
               value <= std::numeric_limits<double>::max();
    }
    static constexpr bool on_bisector(double x, double y) noexcept
    {
        const double scale = x > y ? x : y;
        if (!(scale > 0.0) || !finite_value(scale))
            return false;
        const double nx = x / scale, ny = y / scale;
        return abs_value((nx - ny) - SQRT2 * ny) <= EPS_BOUND * (nx + ny);
    }
    static constexpr bool above_bisector(double x, double y) noexcept
    {
        const double scale = x > y ? x : y;
        if (!(scale > 0.0) || !finite_value(scale))
            return false;
        const double nx = x / scale, ny = y / scale;
        return (nx - ny) < (SQRT2 * ny - EPS_BOUND * nx);
    }
    static constexpr double rotated_sum(double a, double b) noexcept
    {
        // Nonnegative finite arguments. Avoid a+b overflow before division by sqrt(2).
        const double scale = a > b ? a : b;
        if (scale == 0.0)
            return 0.0;
        return scale * (((a / scale) + (b / scale)) * INV_SQRT2);
    }

    static constexpr D16Complex make(double x, double y, std::uint16_t mask)
    {
        if (!finite_value(x) || !finite_value(y) || x < 0.0 || y < 0.0)
            throw std::invalid_argument("x and y must be finite and nonnegative");
        if (mask == 0x0000u)
        {
            if (x != 0.0 || y != 0.0)
                throw std::invalid_argument("NaN mask requires (0,0)");
            return nan();
        }
        if (x == 0.0 && y == 0.0)
            return zero();
        if (mask == 0xFFFFu)
            throw std::invalid_argument("0xFFFF is reserved for zero");

        if (is_one_hot(mask))
        {
            int k = bit_index(mask);
            // 1. Swap x/y while preserving the value: even k -> k+3, odd k -> k-3 (mod 16).
            if (y > x)
            {
                const double tmp = x;
                x = y;
                y = tmp;
                k = (k + ((k & 1) ? 13 : 3)) & 15;
            }
            // 2. Fold across the 22.5° bisector if above_bisector(x, y) (swaps 0<->1 and 2<->3)
            if (above_bisector(x, y))
            {
                const double nx = rotated_sum(x, y);
                const double ny = (x - y) * INV_SQRT2;
                x = nx;
                y = ny;
                k ^= 1;
            }
            return reduced(x, y, static_cast<std::uint16_t>(1u << k));
        }

        int bit_count = 0;
        for (int b = 0; b < 16; ++b)
            bit_count += (mask >> b) & 1u;
        const bool on_y0 = (y <= EPS_BOUND * x);
        const bool on_bis = !on_y0 && on_bisector(x, y);
        if (bit_count != 2 || y > x || (!on_y0 && !on_bis))
            throw std::invalid_argument("invalid boundary representation");

        const int k = even_anchor(mask);
        if (k < 0)
            throw std::invalid_argument("boundary mask incompatible with coordinates");
        const int neighbor = on_y0 ? ((k + 15) & 15) : ((k + 1) & 15);
        const auto expected = static_cast<std::uint16_t>((1u << k) | (1u << neighbor));
        if (mask != expected)
            throw std::invalid_argument("boundary mask incompatible with coordinates");

        return {x, on_y0 ? 0.0 : (TAN_PI_8 * x), mask};
    }

    static constexpr D16Complex reduced(double x, double y, std::uint16_t counter) noexcept
    {
        if (!finite_value(x) || !finite_value(y))
            return nan();
        if (x == 0.0 && y == 0.0)
            return zero();
        const int k = bit_index(counter);
        if (y <= EPS_BOUND * x)
        {
            y = 0.0;
            const int neighbor = (k & 1) ? ((k + 1) & 15) : ((k + 15) & 15);
            counter = static_cast<std::uint16_t>(counter | (1u << neighbor));
        }
        else if (on_bisector(x, y))
        {
            y = TAN_PI_8 * x;
            const int neighbor = (k & 1) ? ((k + 15) & 15) : ((k + 1) & 15);
            counter = static_cast<std::uint16_t>(counter | (1u << neighbor));
        }
        return {x, y, counter};
    }

    // Pure bitwise group action of Z_8 |>< Z_2 (op_id in 0..15): (x, y) NEVER changes!
    constexpr D16Complex apply_group_op(unsigned op_id) const noexcept
    {
        if (is_zero() || is_nan())
            return *this;
        const std::uint16_t base_m = (op_id < 8u) ? rotation : reverse_bits(rotation);
        const unsigned shift = 2u * (op_id & 7u);
        return {x, y, rol(base_m, shift)};
    }

    static D16Complex from_xy(double logic, double emotion) noexcept
    {
        if (!std::isfinite(logic) || !std::isfinite(emotion))
            return nan();
        if (logic == 0.0 && emotion == 0.0)
            return zero();

        int q = 0;
        double u = 0.0, v = 0.0;
        if (logic > 0.0 && emotion >= 0.0)
        {
            q = 0;
            u = logic;
            v = emotion;
        }
        else if (logic <= 0.0 && emotion > 0.0)
        {
            q = 1;
            u = emotion;
            v = -logic;
        }
        else if (logic < 0.0 && emotion <= 0.0)
        {
            q = 2;
            u = -logic;
            v = -emotion;
        }
        else
        {
            q = 3;
            u = -emotion;
            v = logic;
        }

        // Step 1: fold into 45° octant (0 <= wy <= wx)
        const bool odd_oct = (v >= u);
        const double wx = odd_oct ? v : u;
        const double wy = odd_oct ? u : v;

        // Step 2: fold into 22.5° fundamental wedge (0 <= y <= (sqrt(2)-1)*x)
        // Condition wy > (sqrt(2)-1)*wx is equivalent to (wx - wy) < sqrt(2)*wy
        const bool near_diag = above_bisector(wx, wy);
        double x, y;
        int s16;
        if (!near_diag)
        {
            x = wx;
            y = wy;
            s16 = 4 * q + (odd_oct ? 3 : 0);
        }
        else
        {
            x = rotated_sum(wx, wy);
            y = (wx - wy) * INV_SQRT2;
            s16 = 4 * q + (odd_oct ? 2 : 1);
        }
        return reduced(x, y, static_cast<std::uint16_t>(1u << s16));
    }

    std::pair<double, double> to_xy() const noexcept
    {
        if (is_zero())
            return {0.0, 0.0};
        if (is_nan())
            return {NAN, NAN};
        const int s = state();
        const int q = s >> 2;
        const int r = s & 3;

        // Unfold from 22.5° wedge to 45° octant
        const bool is_diag_sector = (r == 1 || r == 2);
        const double wx = is_diag_sector ? (rotated_sum(x, y)) : x;
        const double wy = is_diag_sector ? ((x - y) * INV_SQRT2) : y;

        // Unfold from 45° octant to quadrant
        const double u = (r >= 2) ? wy : wx;
        const double v = (r >= 2) ? wx : wy;

        switch (q)
        {
        case 0:
            return {u, v};
        case 1:
            return {-v, u};
        case 2:
            return {-u, -v};
        default:
            return {v, -u};
        }
    }

    D16Complex multiply(const D16Complex &other) const noexcept
    {
        if (is_nan() || other.is_nan())
            return nan();
        if (is_zero() || other.is_zero())
            return zero();

        const int ka = even_anchor(rotation);
        const int kb = even_anchor(other.rotation);
        const int ma = ((ka + 1) / 2) & 7; // nearest 45° ray index in 0..7
        const int mb = ((kb + 1) / 2) & 7;
        const int pa = ka & 1; // 0: +alpha, 1: -alpha
        const int pb = kb & 1;

        const double ac = x * other.x;
        const double bd = y * other.y;
        const double ad = x * other.y;
        const double bc = y * other.x;

        std::uint16_t counter = rol(0x0001u, static_cast<unsigned>(2 * ((ma + mb) & 7)));
        double out_x, out_y;

        if (pa == pb)
        {
            // Angles add: alpha + beta in [0°, 45°]
            const double u = ac - bd;
            const double v = ad + bc;
            // Crosses 22.5° bisector iff v > (sqrt(2)-1)*u <=> (u - v) < sqrt(2)*v
            const bool cross_bisector = above_bisector(u, v);
            if (pa == 0)
            {
                if (cross_bisector)
                    counter = rol(counter, 1);
            }
            else
            {
                counter = ror(counter, 1);
                if (cross_bisector)
                    counter = ror(counter, 1);
            }
            if (cross_bisector)
            {
                out_x = rotated_sum(u, v);
                out_y = (u - v) * INV_SQRT2;
            }
            else
            {
                out_x = u;
                out_y = v;
            }
        }
        else
        {
            // Angles subtract: |alpha - beta| in [0°, 22.5°] (never crosses 22.5°!)
            const bool negative_deviation = (pa == 0) ? (bc < ad) : (ad < bc);
            if (negative_deviation)
                counter = ror(counter, 1);
            out_x = ac + bd;
            out_y = (bc >= ad) ? (bc - ad) : (ad - bc);
        }
        if (!std::isfinite(out_x) || !std::isfinite(out_y))
            return nan();
        return reduced(out_x, out_y, counter);
    }

    D16Complex operator*(const D16Complex &rhs) const noexcept { return multiply(rhs); }
};