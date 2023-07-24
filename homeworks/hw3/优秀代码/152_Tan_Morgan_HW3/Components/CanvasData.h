#pragma once

#include <UGM/UGM.h>

struct CanvasData {
	std::vector<Ubpa::pointf2> points;
	Ubpa::valf2 scrolling{ 0.f,0.f };
	bool opt_enable_grid{ true };
	bool opt_enable_context_menu{ true };
	bool adding_point{ true };
	std::vector<std::vector<Ubpa::pointf2>> interp_points;
	std::array<bool, 4> update_interp{ false };
	std::array<bool, 4> opt_enable_param{ false };
};

#include "details/CanvasData_AutoRefl.inl"
