#include "CanvasSystem.h"

#include "../Components/CanvasData.h"

#include <_deps/imgui/imgui.h>

#include <Eigen/Dense>

using namespace Ubpa;


void Gaussian(const std::vector<float> &xs, std::vector<float> ys, std::vector<float>& ret, int n_ret=-1)
{
	int n = (int)xs.size();
	int m = 2;
	
	float sigma2 = 1.0;
	Eigen::MatrixXf J(n, m+n);
	Eigen::VectorXf v(n);
	J.setZero();
	for (int i = 0; i < n; ++i)
	{
		v(i) = ys[i];
		for (int j = m; j < m+n; ++j)
			J(i, j) = std::exp(-0.5 * std::pow(xs[j-m] - xs[i], 2) / sigma2);
		for (int j = 0; j < m; ++j)
			J(i, j) = std::pow(xs[i], j);
	}
	Eigen::VectorXf b = J.colPivHouseholderQr().solve(v);

	auto interp = [&](float x) {
		float y = b(0);
		for (int j = 1; j < m; ++j)
			y += b(j) * std::pow(x, j);
		for (int j = m; j < m+n; ++j)
			y += b(j) * std::exp(-0.5 * std::pow(x - xs[j-m], 2) / sigma2);
		return y;
	};

	float xmax = *std::max_element(xs.begin(), xs.end());
	float xmin = *std::min_element(xs.begin(), xs.end());
	int k = n_ret;
	if (n_ret == -1)
		k = std::max(int(xmax - xmin) + 1, 2 * n);
	float d = (xmax - xmin) / (float)(k - 1);

	ret.resize(k);
	for (int i = 0; i < k; ++i)
	{
		ret[i] = interp(xmin + d * i);
	}

}


/* mode = [1, 2, 3, 4]
* 1. Uniform 2. Chord 3. Centripetal 4. Foley
*/
void CurveParameterize(const std::vector<pointf2> points, int mode, std::vector<pointf2>& ret)
{
	size_t n = points.size();
	std::vector<float> t(n, 0.0);
	std::vector<float> d(n, 0.0);
	std::vector<float> a;
	float dt = 1.0f / (float)(n - 1);
	if (mode > 1)
	{
		for (size_t i = 1; i < n; ++i)
			d[i] = (points[i] - points[i - 1]).norm();
	}

	switch (mode)
	{
	case 1:
		for (size_t i = 1; i < n; ++i)
			t[i] = t[i - 1] + dt;
		break;
	case 2:
		for (size_t i = 1; i < n; ++i)
			t[i] += t[i - 1] + d[i];
		for (size_t i = 1; i < n; ++i)
			t[i] /= t.back();
		break;
	case 3:
		for (size_t i = 1; i < n; ++i)
			t[i] += t[i - 1] + std::sqrt(d[i]);
		for (size_t i = 1; i < n; ++i)
			t[i] /= t.back();
		break;
	case 4:
		a.resize(n);
		a[0] = 0;
		for (size_t i = 1; i < n - 1; ++i)
		{
			float ai = std::acos((points[i] - points[i - 1]).normalize().dot((
				points[i] - points[i + 1]).normalize()));
			a[i] = std::min(PI<float> -ai, PI<float> / 2);
		}
		for (size_t i = 1; i < n; ++i)
		{
			float b1, b2;
			b1 = i < n - 1 ? a[i] * d[i] / (d[i] + d[i + 1]) : 0.0f;
			b2 = i < n - 2 ? a[i + 1] * d[i + 1] / (d[i + 1] + d[i + 2]) : 0.0f;
			//b1 = i < n - 1 ? a[i] * d[i] / (d[i] + d[i + 1]) : a[i];
			//if (i < n - 2)
			//	b2 = a[i + 1] * d[i + 1] / (d[i + 1] + d[i + 2]);
			//else if (i == n - 2)
			//	b2 = a[i + 1];
			//else
			//	b2 = 0.0f;
			t[i] = t[i - 1] + d[i] * (1.0f + 1.5f * b1 + 1.5f * b2);
		}
		for (size_t i = 1; i < n; ++i)
			t[i] /= t.back();
		break;
	default:
		break;
	}
	
	std::vector<float> xs(n), ys(n);
	std::vector<float> retx, rety;
	for (size_t i = 0; i < n; ++i)
	{
		xs[i] = points[i][0];
		ys[i] = points[i][1];
	}
	float xmax = *std::max_element(xs.begin(), xs.end());
	float xmin = *std::min_element(xs.begin(), xs.end());
	int n_ret = std::max(int((xmax - xmin) / 0.5f), 2 * (int)n) + 1;
	Gaussian(t, xs, retx, n_ret);
	Gaussian(t, ys, rety, n_ret);
	ret.resize(retx.size());
	for (size_t i = 0; i < ret.size(); ++i)
	{
		ret[i] = pointf2(retx[i], rety[i]);
	}
}


void Plot(ImDrawList* draw_list, const ImVec2& origin, 
	const std::vector<pointf2>& points, const ImU32 color, const float width)
{
	if (points.empty())
		return;

	for (size_t i = 0; i < points.size() - 1; i++)
		draw_list->AddLine(
			ImVec2(origin.x + points[i][0], origin.y + points[i][1]),
			ImVec2(origin.x + points[i + 1][0], origin.y + points[i + 1][1]),
			color, width
		);
}


void CanvasSystem::OnUpdate(Ubpa::UECS::Schedule& schedule) {
	schedule.RegisterCommand([](Ubpa::UECS::World* w) {
		auto data = w->entityMngr.GetSingleton<CanvasData>();

		// resize interp_points
		const size_t n_interps = 4;
		if (data->interp_points.size() != n_interps)
		{
			data->interp_points.resize(n_interps);
		}

		if (!data)
			return;
		
		if (ImGui::Begin("Canvas")) 
		{
			std::string paramaterize_type[n_interps] = {
				"Unifrom", "Chord", "Centripetal", "Foley" 
			};
			std::vector<bool> check_box_state(n_interps);
			ImGui::Checkbox("Enable grid", &data->opt_enable_grid);
			ImGui::Checkbox("Enable context menu", &data->opt_enable_context_menu);
			for (int i = 0; i < n_interps; ++i)
				check_box_state[i] = ImGui::Checkbox(paramaterize_type[i].c_str(), &data->opt_enable_param[i]);
			ImGui::Text("Mouse Left: drag to add lines,\nMouse Right: drag to scroll, click for context menu.");

			for (size_t i = 0; i < n_interps; ++i)
				data->update_interp[i] = check_box_state[i] && data->interp_points[i].empty();

			// Using InvisibleButton() as a convenience 1) it will advance the layout cursor and 2) allows us to use IsItemHovered()/IsItemActive()
			ImVec2 canvas_p0 = ImGui::GetCursorScreenPos();      // ImDrawList API uses screen coordinates!
			ImVec2 canvas_sz = ImGui::GetContentRegionAvail();   // Resize canvas to what's available
			if (canvas_sz.x < 50.0f) canvas_sz.x = 50.0f;
			if (canvas_sz.y < 50.0f) canvas_sz.y = 50.0f;
			ImVec2 canvas_p1 = ImVec2(canvas_p0.x + canvas_sz.x, canvas_p0.y + canvas_sz.y);

			// Draw border and background color
			ImGuiIO& io = ImGui::GetIO();
			ImDrawList* draw_list = ImGui::GetWindowDrawList();
			draw_list->AddRectFilled(canvas_p0, canvas_p1, IM_COL32(50, 50, 50, 255));
			draw_list->AddRect(canvas_p0, canvas_p1, IM_COL32(255, 255, 255, 255));

			// This will catch our interactions
			ImGui::InvisibleButton("canvas", canvas_sz, ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
			const bool is_hovered = ImGui::IsItemHovered(); // Hovered
			const bool is_active = ImGui::IsItemActive();   // Held
			const ImVec2 origin(canvas_p0.x + data->scrolling[0], canvas_p0.y + data->scrolling[1]); // Lock scrolled origin
			const pointf2 mouse_pos_in_canvas(io.MousePos.x - origin.x, io.MousePos.y - origin.y);

			bool is_mouse_in_canvas = mouse_pos_in_canvas[0] >= 0 && mouse_pos_in_canvas[1] >= 0
				&& mouse_pos_in_canvas[0] <= canvas_sz.x && mouse_pos_in_canvas[1] < canvas_sz.y;
			if (is_mouse_in_canvas && ImGui::IsMouseClicked(ImGuiMouseButton_Left) && data->adding_point)
			{
				data->points.push_back(mouse_pos_in_canvas);
				std::fill(data->update_interp.begin(), data->update_interp.end(), true);
			}

			// Pan (we use a zero mouse threshold when there's no context menu)
			// You may decide to make that threshold dynamic based on whether the mouse is hovering something etc.
			const float mouse_threshold_for_pan = data->opt_enable_context_menu ? -1.0f : 0.0f;
			if (is_active && ImGui::IsMouseDragging(ImGuiMouseButton_Right, mouse_threshold_for_pan))
			{
				data->scrolling[0] += io.MouseDelta.x;
				data->scrolling[1] += io.MouseDelta.y;
			}

			// Context menu (under default mouse threshold)
			ImVec2 drag_delta = ImGui::GetMouseDragDelta(ImGuiMouseButton_Right);
			if (data->opt_enable_context_menu && ImGui::IsMouseReleased(ImGuiMouseButton_Right) && drag_delta.x == 0.0f && drag_delta.y == 0.0f)
				ImGui::OpenPopupContextItem("context");
			if (ImGui::BeginPopup("context"))
			{
				data->adding_point = false;
				if (ImGui::MenuItem("Remove one", NULL, false, data->points.size() > 0)) {
					data->points.pop_back();
					std::fill(data->update_interp.begin(), data->update_interp.end(), true);
				}
				if (ImGui::MenuItem("Remove all", NULL, false, data->points.size() > 0)) { 
					data->points.clear(); 
					for (auto& v : data->interp_points)
						v.clear();
				}
				ImGui::EndPopup();
			}
			else
			{
				data->adding_point = true;
			}

			// Draw grid + all lines in the canvas
			draw_list->PushClipRect(canvas_p0, canvas_p1, true);
			if (data->opt_enable_grid)
			{
				const float GRID_STEP = 64.0f;
				for (float x = fmodf(data->scrolling[0], GRID_STEP); x < canvas_sz.x; x += GRID_STEP)
					draw_list->AddLine(ImVec2(canvas_p0.x + x, canvas_p0.y), ImVec2(canvas_p0.x + x, canvas_p1.y), IM_COL32(200, 200, 200, 40));
				for (float y = fmodf(data->scrolling[1], GRID_STEP); y < canvas_sz.y; y += GRID_STEP)
					draw_list->AddLine(ImVec2(canvas_p0.x, canvas_p0.y + y), ImVec2(canvas_p1.x, canvas_p0.y + y), IM_COL32(200, 200, 200, 40));
			}
			
			ImU32 colors[n_interps + 1] = {
				IM_COL32(255, 0, 0, 255), IM_COL32(0, 255, 0, 255),
				IM_COL32(0, 0, 255, 255), IM_COL32(255, 255, 0, 255)
			};
			float width = 2.0f;

			// draw points
			for (size_t i = 0; i < data->points.size(); ++i)
			{
				draw_list->AddCircleFilled(
					ImVec2(origin.x + data->points[i][0], origin.y + data->points[i][1]),
					4.0, IM_COL32(255, 255, 255, 255)
				);
			}
			
			if (data->points.size() >= 2)
			{
				for (size_t i = 0; i < data->interp_points.size(); ++i)
				{
					if (data->update_interp[i] && data->opt_enable_param[i])
						CurveParameterize(data->points, i + 1, data->interp_points[i]);

					data->update_interp[i] = false;
				}
			}

			//draw interploation line
			for (int i = 0; i < n_interps; ++i)
			{
				if (data->opt_enable_param[i])
					Plot(draw_list, origin, data->interp_points[i], colors[i], width);
			}

			draw_list->PopClipRect();
		}

		ImGui::End();
	});
}





