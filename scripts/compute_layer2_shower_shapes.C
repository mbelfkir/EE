#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "TCanvas.h"
#include "TFile.h"
#include "TH1D.h"
#include "TLegend.h"
#include "TStyle.h"
#include "TTree.h"

namespace {

constexpr double kMmPerDelta01 = 144.0;
constexpr double kMmPerUnit = kMmPerDelta01 / 0.1;  // 1440 mm per eta/phi unit
constexpr double kCellSize = 0.025;

double x_to_eta(double x_mm) { return x_mm / kMmPerUnit; }
double y_to_phi(double y_mm) { return y_mm / kMmPerUnit; }

struct ShowerShapes {
  std::vector<double> reta;
  std::vector<double> rphi;
  std::vector<double> weta2;
  std::vector<double> wphi2;
  std::vector<double> e1x1_over_e7x7;
  std::vector<double> e3x3_over_e7x7;
  std::vector<double> fside;
};

bool in_window(int deta_bin, int dphi_bin, int neta, int nphi) {
  const int half_eta = (neta - 1) / 2;
  const int half_phi = (nphi - 1) / 2;
  return std::abs(deta_bin) <= half_eta && std::abs(dphi_bin) <= half_phi;
}

ShowerShapes compute_shapes(const std::string& input_path) {
  ShowerShapes result;

  TFile file(input_path.c_str(), "READ");
  if (file.IsZombie()) {
    std::cerr << "Failed to open " << input_path << std::endl;
    return result;
  }

  auto* tree = dynamic_cast<TTree*>(file.Get("physics"));
  if (!tree) {
    std::cerr << "Missing physics tree in " << input_path << std::endl;
    return result;
  }

  std::vector<double>* cell_e = nullptr;
  std::vector<double>* cell_x = nullptr;
  std::vector<double>* cell_y = nullptr;
  std::vector<int>* cell_l = nullptr;

  tree->SetBranchAddress("cell_e", &cell_e);
  tree->SetBranchAddress("cell_x", &cell_x);
  tree->SetBranchAddress("cell_y", &cell_y);
  tree->SetBranchAddress("cell_l", &cell_l);

  result.reta.reserve(tree->GetEntries());
  result.rphi.reserve(tree->GetEntries());
  result.weta2.reserve(tree->GetEntries());

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

    double e3x3 = 0.0;
    double e3x5 = 0.0;
    double e3x7 = 0.0;
    double e7x3 = 0.0;
    double e7x7 = 0.0;
    double e1x1 = 0.0;
    double sum_e_eta = 0.0;
    double sum_e_eta2 = 0.0;
    double sum_e_phi = 0.0;
    double sum_e_phi2 = 0.0;

    for (size_t i = 0; i < cell_l->size(); ++i) {
      if ((*cell_l)[i] != 2) {
        continue;
      }

      const double deta = x_to_eta((*cell_x)[i]) - hot_eta;
      const double dphi = y_to_phi((*cell_y)[i]) - hot_phi;
      const int deta_bin = static_cast<int>(std::llround(deta / kCellSize));
      const int dphi_bin = static_cast<int>(std::llround(dphi / kCellSize));
      const double energy = (*cell_e)[i];

      if (in_window(deta_bin, dphi_bin, 7, 7)) {
        e7x7 += energy;
      }
      if (in_window(deta_bin, dphi_bin, 7, 3)) {
        e7x3 += energy;
      }
      if (in_window(deta_bin, dphi_bin, 3, 7)) {
        e3x7 += energy;
      }
      if (in_window(deta_bin, dphi_bin, 3, 5)) {
        e3x5 += energy;
        sum_e_eta += energy * deta;
        sum_e_eta2 += energy * deta * deta;
        sum_e_phi += energy * dphi;
        sum_e_phi2 += energy * dphi * dphi;
      }
      if (in_window(deta_bin, dphi_bin, 3, 3)) {
        e3x3 += energy;
      }
      if (in_window(deta_bin, dphi_bin, 1, 1)) {
        e1x1 += energy;
      }
    }

    if (e7x7 > 0.0 && e3x7 > 0.0 && e3x5 > 0.0) {
      result.reta.push_back(e3x7 / e7x7);
      result.rphi.push_back(e3x3 / e3x7);
      const double mean_eta = sum_e_eta / e3x5;
      const double mean_eta2 = sum_e_eta2 / e3x5;
      const double variance = std::max(0.0, mean_eta2 - mean_eta * mean_eta);
      result.weta2.push_back(std::sqrt(variance));
      const double mean_phi = sum_e_phi / e3x5;
      const double mean_phi2 = sum_e_phi2 / e3x5;
      const double variance_phi = std::max(0.0, mean_phi2 - mean_phi * mean_phi);
      result.wphi2.push_back(std::sqrt(variance_phi));
      result.e1x1_over_e7x7.push_back(e1x1 / e7x7);
      result.e3x3_over_e7x7.push_back(e3x3 / e7x7);
      result.fside.push_back((e7x3 - e3x3) / e3x7);
    }
  }

  return result;
}

TH1D make_hist(const std::string& name,
               const std::string& title,
               const std::vector<double>& values,
               int nbins,
               double xmin,
               double xmax,
               int color) {
  TH1D hist(name.c_str(), title.c_str(), nbins, xmin, xmax);
  hist.SetDirectory(nullptr);
  hist.SetLineColor(color);
  hist.SetLineWidth(3);
  for (double value : values) {
    hist.Fill(value);
  }
  if (hist.Integral() > 0.0) {
    hist.Scale(1.0 / hist.Integral());
  }
  return hist;
}

void draw_pair(const std::vector<double>& photon_values,
               const std::vector<double>& electron_values,
               const std::string& output_path,
               const std::string& title,
               const std::string& xaxis,
               int nbins,
               double xmin,
               double xmax) {
  auto photon_hist = make_hist("photon_hist", title + ";" + xaxis + ";Normalized entries",
                               photon_values, nbins, xmin, xmax, kRed + 1);
  auto electron_hist = make_hist("electron_hist", title + ";" + xaxis + ";Normalized entries",
                                 electron_values, nbins, xmin, xmax, kBlue + 1);

  const double max_y = std::max(photon_hist.GetMaximum(), electron_hist.GetMaximum());
  photon_hist.SetMaximum(max_y * 1.2);

  TCanvas canvas("canvas", title.c_str(), 900, 700);
  photon_hist.Draw("hist");
  electron_hist.Draw("hist same");

  TLegend legend(0.63, 0.72, 0.88, 0.88);
  legend.SetBorderSize(0);
  legend.AddEntry(&photon_hist, "photon 30 GeV", "l");
  legend.AddEntry(&electron_hist, "electron 30 GeV", "l");
  legend.Draw();

  canvas.SaveAs(output_path.c_str());
  std::cout << "Saved " << output_path << std::endl;
}

}  // namespace

void compute_layer2_shower_shapes(const char* photon_path = "../data/photon_30GeV_10k.root",
                                  const char* electron_path = "../data/electron_30GeV_10k.root",
                                  const char* output_dir = "outputs/plots") {
  std::filesystem::create_directories(output_dir);
  gStyle->SetOptStat(0);

  const auto photon = compute_shapes(photon_path);
  const auto electron = compute_shapes(electron_path);

  draw_pair(photon.reta, electron.reta,
            std::string(output_dir) + "/reta_layer2_30GeV.png",
            "R_{#eta} in layer 2",
            "E_{3x7} / E_{7x7}",
            60, 0.0, 1.1);

  draw_pair(photon.rphi, electron.rphi,
            std::string(output_dir) + "/rphi_layer2_30GeV.png",
            "R_{#phi} in layer 2",
            "E_{3x3} / E_{3x7}",
            60, 0.0, 1.1);

  draw_pair(photon.weta2, electron.weta2,
            std::string(output_dir) + "/weta2_layer2_30GeV.png",
            "W_{#eta,2} in layer 2",
            "W_{#eta,2}",
            60, 0.0, 0.08);

  draw_pair(photon.wphi2, electron.wphi2,
            std::string(output_dir) + "/wphi2_layer2_30GeV.png",
            "W_{#phi,2} in layer 2",
            "W_{#phi,2}",
            60, 0.0, 0.08);

  draw_pair(photon.e1x1_over_e7x7, electron.e1x1_over_e7x7,
            std::string(output_dir) + "/e1x1_over_e7x7_layer2_30GeV.png",
            "E_{1x1} / E_{7x7} in layer 2",
            "E_{1x1} / E_{7x7}",
            60, 0.0, 1.0);

  draw_pair(photon.e3x3_over_e7x7, electron.e3x3_over_e7x7,
            std::string(output_dir) + "/e3x3_over_e7x7_layer2_30GeV.png",
            "E_{3x3} / E_{7x7} in layer 2",
            "E_{3x3} / E_{7x7}",
            60, 0.0, 1.0);

  draw_pair(photon.fside, electron.fside,
            std::string(output_dir) + "/fside_layer2_30GeV.png",
            "F_{side} in layer 2",
            "(E_{7x3} - E_{3x3}) / E_{3x7}",
            60, 0.0, 1.0);

  std::cout << "Photon events used: " << photon.reta.size() << std::endl;
  std::cout << "Electron events used: " << electron.reta.size() << std::endl;
}
