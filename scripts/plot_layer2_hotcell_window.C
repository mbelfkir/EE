#include <algorithm>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "TCanvas.h"
#include "TFile.h"
#include "TH2D.h"
#include "TLine.h"
#include "TStyle.h"
#include "TTree.h"

namespace {

constexpr double kMmPerDelta01 = 144.0;
constexpr double kMmPerUnit = kMmPerDelta01 / 0.1;  // 1440 mm per eta/phi unit

double x_to_eta(double x_mm) { return x_mm / kMmPerUnit; }
double y_to_phi(double y_mm) { return y_mm / kMmPerUnit; }

}  // namespace

void plot_layer2_hotcell_window(const char* input_path,
                                const char* output_path,
                                int nbins_eta = 20,
                                int nbins_phi = 20,
                                double bin_size = 0.025,
                                int inner_nbins_eta = 7,
                                int inner_nbins_phi = 11) {
  TFile file(input_path, "READ");
  if (file.IsZombie()) {
    std::cerr << "Failed to open " << input_path << std::endl;
    return;
  }

  auto* tree = dynamic_cast<TTree*>(file.Get("physics"));
  if (!tree) {
    std::cerr << "Missing physics tree in " << input_path << std::endl;
    return;
  }

  std::vector<double>* cell_e = nullptr;
  std::vector<double>* cell_x = nullptr;
  std::vector<double>* cell_y = nullptr;
  std::vector<int>* cell_l = nullptr;

  tree->SetBranchAddress("cell_e", &cell_e);
  tree->SetBranchAddress("cell_x", &cell_x);
  tree->SetBranchAddress("cell_y", &cell_y);
  tree->SetBranchAddress("cell_l", &cell_l);

  const double eta_half_width = 0.5 * nbins_eta * bin_size;
  const double phi_half_width = 0.5 * nbins_phi * bin_size;

  TH2D hist("layer2_hotcell_window",
            "Layer 2 deposited energy around hottest cell;#Delta#eta from hottest cell;#Delta#phi from hottest cell",
            nbins_eta,
            -eta_half_width,
            eta_half_width,
            nbins_phi,
            -phi_half_width,
            phi_half_width);
  hist.SetDirectory(nullptr);

  const double inner_eta_half_width = 0.5 * inner_nbins_eta * bin_size;
  const double inner_phi_half_width = 0.5 * inner_nbins_phi * bin_size;

  TH2D inner_hist("layer2_hotcell_window_inner",
                  "Layer 2 deposited energy around hottest cell (7 bins in #eta, 11 bins in #phi);#Delta#eta from hottest cell;#Delta#phi from hottest cell",
                  inner_nbins_eta,
                  -inner_eta_half_width,
                  inner_eta_half_width,
                  inner_nbins_phi,
                  -inner_phi_half_width,
                  inner_phi_half_width);
  inner_hist.SetDirectory(nullptr);

  Long64_t used_events = 0;

  for (Long64_t ievt = 0; ievt < tree->GetEntries(); ++ievt) {
    tree->GetEntry(ievt);

    double hot_e = std::numeric_limits<double>::lowest();
    double hot_eta = 0.0;
    double hot_phi = 0.0;
    bool found_hot = false;

    for (size_t i = 0; i < cell_l->size(); ++i) {
      if ((*cell_l)[i] != 2) {
        continue;
      }

      const double energy = (*cell_e)[i];
      if (energy > hot_e) {
        hot_e = energy;
        hot_eta = x_to_eta((*cell_x)[i]);
        hot_phi = y_to_phi((*cell_y)[i]);
        found_hot = true;
      }
    }

    if (!found_hot) {
      continue;
    }

    ++used_events;

    for (size_t i = 0; i < cell_l->size(); ++i) {
      if ((*cell_l)[i] != 2) {
        continue;
      }

      const double deta = x_to_eta((*cell_x)[i]) - hot_eta;
      const double dphi = y_to_phi((*cell_y)[i]) - hot_phi;

      if (deta < -eta_half_width || deta >= eta_half_width ||
          dphi < -phi_half_width || dphi >= phi_half_width) {
        continue;
      }

      hist.Fill(deta, dphi, (*cell_e)[i]);

      if (deta >= -inner_eta_half_width && deta < inner_eta_half_width &&
          dphi >= -inner_phi_half_width && dphi < inner_phi_half_width) {
        inner_hist.Fill(deta, dphi, (*cell_e)[i]);
      }
    }
  }

  if (used_events == 0) {
    std::cerr << "No events with layer-2 cells found in " << input_path << std::endl;
    return;
  }

  hist.Scale(1.0 / static_cast<double>(used_events));
  inner_hist.Scale(1.0 / static_cast<double>(used_events));

  if (hist.Integral() > 0.0) {
    hist.Scale(1.0 / hist.Integral());
  }
  if (inner_hist.Integral() > 0.0) {
    inner_hist.Scale(1.0 / inner_hist.Integral());
  }

  std::filesystem::create_directories(std::filesystem::path(output_path).parent_path());

  gStyle->SetOptStat(0);
  TCanvas canvas("canvas", "Layer 2 hot-cell-centered window", 920, 760);
  canvas.SetRightMargin(0.16);
  hist.Draw("COLZ");

  TLine top(-inner_eta_half_width, inner_phi_half_width, inner_eta_half_width, inner_phi_half_width);
  TLine bottom(-inner_eta_half_width, -inner_phi_half_width, inner_eta_half_width, -inner_phi_half_width);
  TLine left(-inner_eta_half_width, -inner_phi_half_width, -inner_eta_half_width, inner_phi_half_width);
  TLine right(inner_eta_half_width, -inner_phi_half_width, inner_eta_half_width, inner_phi_half_width);
  top.SetLineColor(kBlack);
  bottom.SetLineColor(kBlack);
  left.SetLineColor(kBlack);
  right.SetLineColor(kBlack);
  top.SetLineWidth(2);
  bottom.SetLineWidth(2);
  left.SetLineWidth(2);
  right.SetLineWidth(2);
  top.Draw();
  bottom.Draw();
  left.Draw();
  right.Draw();

  canvas.SaveAs(output_path);

  std::string inner_output_path = output_path;
  const std::string suffix = ".png";
  const std::string inner_suffix = "_inner_7x11.png";
  if (inner_output_path.size() >= suffix.size() &&
      inner_output_path.substr(inner_output_path.size() - suffix.size()) == suffix) {
    inner_output_path.replace(inner_output_path.size() - suffix.size(), suffix.size(), inner_suffix);
  } else {
    inner_output_path += inner_suffix;
  }

  TCanvas inner_canvas("inner_canvas", "Layer 2 hot-cell-centered inner window", 920, 760);
  inner_canvas.SetRightMargin(0.16);
  inner_hist.Draw("COLZ");
  inner_canvas.SaveAs(inner_output_path.c_str());

  std::string output_3d_path = output_path;
  const std::string output_3d_suffix = "_3d.png";
  if (output_3d_path.size() >= suffix.size() &&
      output_3d_path.substr(output_3d_path.size() - suffix.size()) == suffix) {
    output_3d_path.replace(output_3d_path.size() - suffix.size(), suffix.size(), output_3d_suffix);
  } else {
    output_3d_path += output_3d_suffix;
  }

  std::string inner_output_3d_path = inner_output_path;
  if (inner_output_3d_path.size() >= suffix.size() &&
      inner_output_3d_path.substr(inner_output_3d_path.size() - suffix.size()) == suffix) {
    inner_output_3d_path.replace(inner_output_3d_path.size() - suffix.size(), suffix.size(), output_3d_suffix);
  } else {
    inner_output_3d_path += output_3d_suffix;
  }

  TCanvas canvas_3d("canvas_3d", "Layer 2 hot-cell-centered window 3D", 980, 800);
  canvas_3d.SetRightMargin(0.14);
  canvas_3d.SetTheta(28);
  canvas_3d.SetPhi(38);
  hist.SetLineColor(kBlack);
  hist.SetLineWidth(1);
  hist.Draw("LEGO2");
  canvas_3d.SaveAs(output_3d_path.c_str());

  TCanvas inner_canvas_3d("inner_canvas_3d", "Layer 2 hot-cell-centered inner window 3D", 980, 800);
  inner_canvas_3d.SetRightMargin(0.14);
  inner_canvas_3d.SetTheta(28);
  inner_canvas_3d.SetPhi(38);
  inner_hist.SetLineColor(kBlack);
  inner_hist.SetLineWidth(1);
  inner_hist.Draw("LEGO2");
  inner_canvas_3d.SaveAs(inner_output_3d_path.c_str());

  std::cout << "Used events: " << used_events << std::endl;
  std::cout << "Saved " << output_path << std::endl;
  std::cout << "Saved " << inner_output_path << std::endl;
  std::cout << "Saved " << output_3d_path << std::endl;
  std::cout << "Saved " << inner_output_3d_path << std::endl;
}
