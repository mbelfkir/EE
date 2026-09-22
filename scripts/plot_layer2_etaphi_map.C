#include <algorithm>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "TCanvas.h"
#include "TFile.h"
#include "TH2D.h"
#include "TStyle.h"
#include "TTree.h"

namespace {

constexpr double kMmPerDelta01 = 144.0;
constexpr double kMmPerUnit = kMmPerDelta01 / 0.1;  // 1440 mm per unit of eta or phi

double x_to_eta(double x_mm) { return x_mm / kMmPerUnit; }
double y_to_phi(double y_mm) { return y_mm / kMmPerUnit; }

}  // namespace

void plot_layer2_etaphi_map(const char* input_path,
                            const char* output_path,
                            int nbins_eta = 80,
                            int nbins_phi = 80) {
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

  double eta_min = std::numeric_limits<double>::max();
  double eta_max = std::numeric_limits<double>::lowest();
  double phi_min = std::numeric_limits<double>::max();
  double phi_max = std::numeric_limits<double>::lowest();
  bool found_layer2 = false;

  for (Long64_t ievt = 0; ievt < tree->GetEntries(); ++ievt) {
    tree->GetEntry(ievt);
    for (size_t i = 0; i < cell_l->size(); ++i) {
      if ((*cell_l)[i] != 2) {
        continue;
      }
      found_layer2 = true;
      const double eta = x_to_eta((*cell_x)[i]);
      const double phi = y_to_phi((*cell_y)[i]);
      eta_min = std::min(eta_min, eta);
      eta_max = std::max(eta_max, eta);
      phi_min = std::min(phi_min, phi);
      phi_max = std::max(phi_max, phi);
    }
  }

  if (!found_layer2) {
    std::cerr << "No cells found with cell_l == 2 in " << input_path << std::endl;
    return;
  }

  const double eta_pad = 0.05 * (eta_max - eta_min);
  const double phi_pad = 0.05 * (phi_max - phi_min);

  TH2D hist("layer2_etaphi_map",
            "Average deposited energy in calorimeter layer 2;#eta (local, centered at 0);#phi (local, centered at 0)",
            nbins_eta,
            eta_min - eta_pad,
            eta_max + eta_pad,
            nbins_phi,
            phi_min - phi_pad,
            phi_max + phi_pad);
  hist.SetDirectory(nullptr);

  const double norm = 1.0 / static_cast<double>(tree->GetEntries());
  for (Long64_t ievt = 0; ievt < tree->GetEntries(); ++ievt) {
    tree->GetEntry(ievt);
    for (size_t i = 0; i < cell_l->size(); ++i) {
      if ((*cell_l)[i] != 2) {
        continue;
      }
      hist.Fill(x_to_eta((*cell_x)[i]), y_to_phi((*cell_y)[i]), (*cell_e)[i] * norm);
    }
  }

  std::filesystem::create_directories(std::filesystem::path(output_path).parent_path());

  gStyle->SetOptStat(0);
  TCanvas canvas("canvas", "Layer 2 eta-phi map", 920, 760);
  canvas.SetRightMargin(0.16);
  hist.Draw("COLZ");
  canvas.SaveAs(output_path);

  std::cout << "Saved " << output_path << std::endl;
}
