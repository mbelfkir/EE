#include <filesystem>
#include <iostream>
#include <memory>
#include <limits>
#include <string>
#include <vector>

#include "TCanvas.h"
#include "TFile.h"
#include "TLegend.h"
#include "TH1D.h"
#include "TStyle.h"
#include "TTree.h"

namespace {

struct Sample {
  std::string label;
  std::string path;
  int color;
};

struct ExpressionData {
  std::vector<double> values;
  double min = std::numeric_limits<double>::max();
  double max = std::numeric_limits<double>::lowest();
};

ExpressionData read_expression(TTree* tree, const std::string& expression) {
  ExpressionData data;
  tree->Draw(expression.c_str(), "", "goff");

  const Long64_t n = tree->GetSelectedRows();
  const double* values = tree->GetV1();
  data.values.reserve(n);
  for (Long64_t i = 0; i < n; ++i) {
    const double value = values[i];
    data.values.push_back(value);
    data.min = std::min(data.min, value);
    data.max = std::max(data.max, value);
  }

  if (data.values.empty()) {
    data.min = 0.0;
    data.max = 1.0;
  }

  return data;
}

TH1D* make_histogram(const std::vector<double>& values,
                     const std::string& hist_name,
                     const std::string& hist_title,
                     int bins,
                     double xmin,
                     double xmax,
                     int color) {
  auto* hist = new TH1D(hist_name.c_str(), hist_title.c_str(), bins, xmin, xmax);
  hist->SetDirectory(nullptr);
  hist->SetLineColor(color);
  hist->SetLineWidth(3);

  for (double value : values) {
    hist->Fill(value);
  }

  if (hist->Integral() > 0.0) {
    hist->Scale(1.0 / hist->Integral());
  }

  return hist;
}

void draw_comparison(const std::vector<Sample>& samples,
                     const std::string& branch,
                     const std::string& title,
                     const std::string& xaxis,
                     const std::string& output_path,
                     int bins) {
  TCanvas canvas("canvas", title.c_str(), 900, 700);
  TLegend legend(0.62, 0.68, 0.88, 0.88);
  legend.SetBorderSize(0);

  std::vector<std::unique_ptr<TFile>> files;
  std::vector<std::unique_ptr<TH1D>> histograms;
  std::vector<ExpressionData> datasets;
  double max_y = 0.0;
  double global_min = std::numeric_limits<double>::max();
  double global_max = std::numeric_limits<double>::lowest();

  for (const auto& sample : samples) {
    files.emplace_back(TFile::Open(sample.path.c_str(), "READ"));
    if (!files.back() || files.back()->IsZombie()) {
      std::cerr << "Failed to open " << sample.path << std::endl;
      return;
    }

    auto* tree = dynamic_cast<TTree*>(files.back()->Get("physics"));
    if (!tree) {
      std::cerr << "Missing physics tree in " << sample.path << std::endl;
      return;
    }

    auto data = read_expression(tree, branch);
    global_min = std::min(global_min, data.min);
    global_max = std::max(global_max, data.max);
    datasets.push_back(std::move(data));
  }

  if (!(global_max > global_min)) {
    global_min -= 1.0;
    global_max += 1.0;
  }

  const double padding = 0.05 * (global_max - global_min);
  const double xmin = global_min - padding;
  const double xmax = global_max + padding;

  for (size_t i = 0; i < samples.size(); ++i) {
    auto hist = std::unique_ptr<TH1D>(make_histogram(
        datasets[i].values,
        "hist_" + std::to_string(i) + "_" + branch,
        title + ";" + xaxis + ";Normalized entries",
        bins,
        xmin,
        xmax,
        samples[i].color));

    max_y = std::max(max_y, hist->GetMaximum());
    legend.AddEntry(hist.get(), samples[i].label.c_str(), "l");
    histograms.push_back(std::move(hist));
  }

  for (size_t i = 0; i < histograms.size(); ++i) {
    histograms[i]->SetMaximum(max_y * 1.20);
    histograms[i]->Draw(i == 0 ? "hist" : "hist same");
  }

  legend.Draw();
  canvas.SaveAs(output_path.c_str());
  std::cout << "Saved " << output_path << std::endl;
}

}  // namespace

void plot_basic_shower_comparison(const char* data_dir = "../data",
                                  const char* output_dir = "outputs/plots") {
  gStyle->SetOptStat(0);
  std::filesystem::create_directories(output_dir);

  const std::string base = data_dir;
  const std::vector<Sample> samples = {
      {"electron 30 GeV", base + "/electron_30GeV_10k.root", kBlue + 1},
      {"photon 30 GeV", base + "/photon_30GeV_10k.root", kRed + 1},
      {"electron 50 GeV", base + "/electron_50GeV_10k.root", kAzure + 7},
      {"photon 50 GeV", base + "/photon_50GeV_10k.root", kOrange + 7},
  };

  draw_comparison(samples,
                  "Caltotal_e",
                  "Total deposited energy",
                  "Deposited energy",
                  std::string(output_dir) + "/caltotal_comparison.png",
                  80);

  draw_comparison(samples,
                  "ECAL1_e + ECAL2_e + ECAL3_e",
                  "ECAL deposited energy",
                  "ECAL energy",
                  std::string(output_dir) + "/ecal_total_comparison.png",
                  80);

  draw_comparison(samples,
                  "HCAL1_e + HCAL2_e + HCAL3_e",
                  "HCAL deposited energy",
                  "HCAL energy",
                  std::string(output_dir) + "/hcal_total_comparison.png",
                  80);
}
