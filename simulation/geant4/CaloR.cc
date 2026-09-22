//
// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************
//
// Basic tutorial: https://www.ge.infn.it/geant4/training/ornl_2008/retrievinginformationfromkernel.pdf
/// \file CaloR.cc
/// \brief Main program of the CaloR example
//
//
#include <ctime>

#include "G4Types.hh"

#include "G4RunManager.hh"
#include "G4UImanager.hh"

#include "QGSP_FTFP_BERT.hh"
//#include "G4OpticalPhysics.hh"
#include "G4HadronicProcessStore.hh"
#include "G4VisExecutive.hh"
#include "G4UIExecutive.hh"

#include "G4ParticleTable.hh"
#include "G4ParticleDefinition.hh"
#include "G4DecayTable.hh"
#include "G4VDecayChannel.hh"
#include "G4PhaseSpaceDecayChannel.hh"

// package includes
#include "DetectorConstruction.hh"
#include "ActionInitialization.hh"

namespace {
  void PrintUsage() {
    G4cerr << " Usage: " << G4endl;
    G4cerr << " CaloR [-m macro] [-u UIsession] [-s seedNumber]" << G4endl;
  }
}

int main(int argc, char** argv)
{

  // Evaluate arguments
  //
  if (argc > 7 || (argc - 1) % 2 != 0) {
    PrintUsage();
    return 1;
  }

  G4long seed = (long) time(NULL);
  G4String macro;
  G4String session;
  for ( G4int i=1; i<argc; i=i+2 ) {
    if      ( G4String(argv[i]) == "-m" ) macro = argv[i+1];
    else if ( G4String(argv[i]) == "-u" ) session = argv[i+1];
	else if ( G4String(argv[i]) == "-s" ) {
      seed = G4UIcommand::ConvertToInt(argv[i+1]);
	  G4cout << "Using user random seed = " <<  seed  << G4endl;
    }
    else {
      PrintUsage();
      return 1;
    }
  }

  // Detect interactive mode (if no macro provided) and define UI session
  //
  G4UIExecutive* ui = nullptr;
  if ( ! macro.size() ) {
    ui = new G4UIExecutive(argc, argv, session);
  }

  //choose the Random engine and set a random seed
  CLHEP::HepRandom::setTheEngine(new CLHEP::RanecuEngine());
  CLHEP::HepRandom::setTheSeed(seed);
  G4cout << "Seed: " << CLHEP::HepRandom::getTheSeed() << G4endl;

  auto runManager = new G4RunManager;

  // Set mandatory initialization classes

  // Initialize detector geometry
  auto detector = new CaloRDetectorConstruction;
  runManager-> SetUserInitialization(detector);


  // QGSP_FTFP_BERT covers electromagnetic and hadronic shower development.
  G4VUserPhysicsList* physics = new QGSP_FTFP_BERT;
  runManager-> SetUserInitialization(physics);

  // Supress annoying "HADRONIC PROCESSES SUMMARY"
  G4HadronicProcessStore::Instance()->SetVerbose(0);

  //Set Pi0 decay to photon pair
  G4ParticleTable* fParticleTable = G4ParticleTable::GetParticleTable();
  G4ParticleDefinition* fParticleDef = fParticleTable->FindParticle("pi0");
  G4VDecayChannel* fMode =
                  new G4PhaseSpaceDecayChannel("pi0",1,2,"gamma","gamma");
  G4DecayTable* fTable = new G4DecayTable();
  fTable->Insert(fMode);
  fParticleDef->SetDecayTable(fTable);

  // User Action classes
  auto actionInitialization = new CaloRActionInitialization(detector);
  runManager->SetUserInitialization(actionInitialization);

  // Initialize for ParticleGun settings
  runManager-> Initialize();

  // Visualization is needed only for an interactive session.
  G4VisExecutive* visManager = nullptr;
  if (ui) {
    visManager = new G4VisExecutive;
    visManager->Initialize();
    G4cout << G4endl;
  }

 //get the pointer to the User Interface manager
  G4UImanager* UImanager = G4UImanager::GetUIpointer();

  if (!ui) { // batch mode
    G4String command = "/control/execute ";
    UImanager-> ApplyCommand(command+macro);
  } else {  // interactive mode : define UI session

    // interactive mode : define UI session
    UImanager->ApplyCommand("/control/execute init_vis.mac");
    if (ui->IsGUI()) {
      UImanager->ApplyCommand("/control/execute gui.mac");
    }

    ui-> SessionStart();
    delete ui;
  }

  // Free the store: user actions, physics_list and detector_description are
  //                 owned and deleted by the run manager, so they should not
  //                 be deleted in the main() program !

  delete visManager;
  delete runManager;
}
