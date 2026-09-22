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

void plot_layer2_energy_map(const char* input_path,
                            const char* output_path,
                            int nbins_x = 80,
                            int nbins_y = 80) {
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

  double xmin = std::numeric_limits<double>::max();
  double xmax = std::numeric_limits<double>::lowest();
  double ymin = std::numeric_limits<double>::max();
  double ymax = std::numeric_limits<double>::lowest();
  bool found_layer2 = false;

  for (Long64_t ievt = 0; ievt < tree->GetEntries(); ++ievt) {
    tree->GetEntry(ievt);
    for (size_t i = 0; i < cell_l->size(); ++i) {
      if ((*cell_l)[i] != 2) {
        continue;
      }
      found_layer2 = true;
      xmin = std::min(xmin, (*cell_x)[i]);
      xmax = std::max(xmax, (*cell_x)[i]);
      ymin = std::min(ymin, (*cell_y)[i]);
      ymax = std::max(ymax, (*cell_y)[i]);
    }
  }

  if (!found_layer2) {
    std::cerr << "No cells found with cell_l == 2 in " << input_path << std::endl;
    return;
  }

  const double xpad = 0.05 * (xmax - xmin);
  const double ypad = 0.05 * (ymax - ymin);

  TH2D hist("layer2_map",
            "Average deposited energy in calorimeter layer 2;cell x;cell y",
            nbins_x,
            xmin - xpad,
            xmax + xpad,
            nbins_y,
            ymin - ypad,
            ymax + ypad);
  hist.SetDirectory(nullptr);

  const double norm = 1.0 / static_cast<double>(tree->GetEntries());
  for (Long64_t ievt = 0; ievt < tree->GetEntries(); ++ievt) {
    tree->GetEntry(ievt);
    for (size_t i = 0; i < cell_l->size(); ++i) {
      if ((*cell_l)[i] != 2) {
        continue;
      }
      hist.Fill((*cell_x)[i], (*cell_y)[i], (*cell_e)[i] * norm);
    }
  }

  std::filesystem::create_directories(std::filesystem::path(output_path).parent_path());

  gStyle->SetOptStat(0);
  TCanvas canvas("canvas", "Layer 2 energy map", 900, 760);
  canvas.SetRightMargin(0.16);
  hist.Draw("COLZ");
  canvas.SaveAs(output_path);

  std::cout << "Saved " << output_path << std::endl;
}
