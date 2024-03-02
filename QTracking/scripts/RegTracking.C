#include <cstdlib>
#include <iostream>
#include <map>
#include <string>

#include "TChain.h"
#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "TObjString.h"
#include "TSystem.h"
#include "TROOT.h"

#include "TMVA/Tools.h"
#include "TMVA/Factory.h"
#include "TMVA/DataLoader.h"
#include "TMVA/TMVARegGui.h"


using namespace TMVA;

void RegTracking( TString myMethodList = "" )
{

   TMVA::Tools::Instance();



   // Default MVA methods to be trained + tested
   std::map<std::string,int> Use;

   // Mutidimensional likelihood and Nearest-Neighbour methods
   Use["PDERS"]           = 0;
   Use["PDEFoam"]         = 0;
   Use["KNN"]             = 0;
   //
   // Linear Discriminant Analysis
   Use["LD"]		        = 0;
   //
   // Function Discriminant analysis
   Use["FDA_GA"]          = 0;
   Use["FDA_MC"]          = 0;
   Use["FDA_MT"]          = 0;
   Use["FDA_GAMT"]        = 0;
   //
   // Neural Network
   Use["MLP"]             = 1;
#ifdef R__HAS_TMVACPU
   Use["DNN_CPU"] = 0;
#else
   Use["DNN_CPU"] = 0;
#endif
   //
   // Support Vector Machine
   Use["SVM"]             = 0;
   //
   // Boosted Decision Trees
   Use["BDT"]             = 0;
   Use["BDTG"]            = 0;
   // ---------------------------------------------------------------

   std::cout << std::endl;
   std::cout << "==> Start TMVARegression" << std::endl;

   // Select methods (don't look at this code - not of interest)
   if (myMethodList != "") {
      for (std::map<std::string,int>::iterator it = Use.begin(); it != Use.end(); it++) it->second = 0;

      std::vector<TString> mlist = gTools().SplitString( myMethodList, ',' );
      for (UInt_t i=0; i<mlist.size(); i++) {
         std::string regMethod(mlist[i].Data());

         if (Use.find(regMethod) == Use.end()) {
            std::cout << "Method \"" << regMethod << "\" not known in TMVA under this name. Choose among the following:" << std::endl;
            for (std::map<std::string,int>::iterator it = Use.begin(); it != Use.end(); it++) std::cout << it->first << " ";
            std::cout << std::endl;
            return;
         }
         Use[regMethod] = 1;
      }
   }

   // --------------------------------------------------------------------------------------------------

   // Here the preparation phase begins

   // Create a new root output file
   TString outfileName( "analysis.root" );
   TFile* outputFile = TFile::Open( outfileName, "RECREATE" );

   // Create the factory object. Later you can choose the methods
   // whose performance you'd like to investigate. The factory will
   // then run the performance analysis for you.
   //
   // The first argument is the base of the name of all the
   // weightfiles in the directory weight/
   //
   // The second argument is the output file for the training results
   // All TMVA output can be suppressed by removing the "!" (not) in
   // front of the "Silent" argument in the option string
   TMVA::Factory *factory = new TMVA::Factory( "TMVARegression", outputFile,
                                               "!V:!Silent:Color:DrawProgressBar:AnalysisType=Regression" );


   TMVA::DataLoader *dataloader=new TMVA::DataLoader("dataset");
   // If you wish to modify default settings
   // (please check "src/Config.h" to see all available global options)
   //
   //     (TMVA::gConfig().GetVariablePlotting()).fTimesRMS = 8.0;
   //     (TMVA::gConfig().GetIONames()).fWeightFileDir = "myWeightDirectory";

   // Define the input variables that shall be used for the MVA training
   // note that you may also use variable expressions, such as: "3*var1/var2*abs(var3)"
   // [all types of expressions that can also be parsed by TTree::Draw( "expression" )]

   dataloader->AddVariable("d0X_x","d0X_x", "units", 'F' );
   dataloader->AddVariable("d0X_y","d0X_y", "units", 'F' );
   dataloader->AddVariable("d0Xp_x","d0Xp_x", "units", 'F' );
   dataloader->AddVariable("d0Xp_y","d0Xp_y", "units", 'F' );
   dataloader->AddVariable("d0U_x","d0U_x", "units", 'F' );
   dataloader->AddVariable("d0U_y","d0U_y", "units", 'F' );
   dataloader->AddVariable("d0Up_x","d0Up_x", "units", 'F' );
   dataloader->AddVariable("d0Up_y","d0Up_y", "units", 'F' );
   dataloader->AddVariable("d0V_x","d0V_x", "units", 'F' );
   dataloader->AddVariable("d0V_y","d0V_y", "units", 'F' );
   dataloader->AddVariable("d0Vp_x","d0Vp_x", "units", 'F' );
   dataloader->AddVariable("d0Vp_y","d0Vp_y", "units", 'F' );
   dataloader->AddVariable("d2X_y","d2X_y", "units", 'F' );
   dataloader->AddVariable("d2X_x","d2X_x", "units", 'F' );
   dataloader->AddVariable("d2Xp_y","d2Xp_y", "units", 'F' );
   dataloader->AddVariable("d2Xp_x","d2Xp_x", "units", 'F' );
   dataloader->AddVariable("d2U_y","d2U_y", "units", 'F' );
   dataloader->AddVariable("d2U_x","d2U_x", "units", 'F' );
   dataloader->AddVariable("d2Up_y","d2Up_y", "units", 'F' );
   dataloader->AddVariable("d2Up_x","d2Up_x", "units", 'F' );
   dataloader->AddVariable("d2V_y","d2V_y", "units", 'F' );
   dataloader->AddVariable("d2V_x","d2V_x", "units", 'F' );
   dataloader->AddVariable("d2Vp_y","d2Vp_y", "units", 'F' );
   dataloader->AddVariable("d2Vp_x","d2Vp_x", "units", 'F' );
   dataloader->AddVariable("d3pX_X","d3pX_X", "units", 'F' );
   dataloader->AddVariable("d3pX_Y","d3pX_Y", "units", 'F' );
   dataloader->AddVariable("d3pXp_X","d3pXp_X", "units", 'F' );
   dataloader->AddVariable("d3pXp_Y","d3pXp_Y", "units", 'F' );
   dataloader->AddVariable("d3pU_X","d3pU_X", "units", 'F' );
   dataloader->AddVariable("d3pU_Y","d3pU_Y", "units", 'F' );
   dataloader->AddVariable("d3pUp_X","d3pUp_X", "units", 'F' );
   dataloader->AddVariable("d3pUp_Y","d3pUp_Y", "units", 'F' );
   dataloader->AddVariable("d3pV_X","d3pV_X", "units", 'F' );
   dataloader->AddVariable("d3pV_Y","d3pV_Y", "units", 'F' );
   dataloader->AddVariable("d3pVp_X","d3pVp_X", "units", 'F' );
   dataloader->AddVariable("d3pVp_Y","d3pVp_Y", "units", 'F' );
   dataloader->AddVariable("d3mX_X","d3mX_X", "units", 'F' );
   dataloader->AddVariable("d3mX_Y","d3mX_Y", "units", 'F' );
   dataloader->AddVariable("d3mXp_X","d3mXp_X", "units", 'F' );
   dataloader->AddVariable("d3mXp_Y","d3mXp_Y", "units", 'F' );
   dataloader->AddVariable("d3mU_X","d3mU_X", "units", 'F' );
   dataloader->AddVariable("d3mU_Y","d3mU_Y", "units", 'F' );
   dataloader->AddVariable("d3mUp_X","d3mUp_X", "units", 'F' );
   dataloader->AddVariable("d3mUp_Y","d3mUp_Y", "units", 'F' );
   dataloader->AddVariable("d3mV_X","d3mV_X", "units", 'F' );
   dataloader->AddVariable("d3mV_Y","d3mV_Y", "units", 'F' );
   dataloader->AddVariable("d3mVp_X","d3mVp_X", "units", 'F' );
   dataloader->AddVariable("d3mVp_Y","d3mVp_Y", "units", 'F' );
  
   dataloader->AddSpectator( "Gpx",  "Gpx", "units", 'F' );
   dataloader->AddSpectator( "Gpy",  "Gpy", "units", 'F' );   
   dataloader->AddSpectator( "Gpz",  "Gpz", "units", 'F' );    
      
   // Add the variable carrying the regression target
   dataloader->AddTarget( "Gpx" );
   dataloader->AddTarget( "Gpy" );
   dataloader->AddTarget( "Gpz" );


   // It is also possible to declare additional targets for multi-dimensional regression, ie:
   //     factory->AddTarget( "fvalue2" );
   // BUT: this is currently ONLY implemented for MLP

   // Read training and test data (see TMVAClassification for reading ASCII files)
   // load the signal and background event samples from ROOT trees
   TFile *input(0);
   TString fname = "./TreeR_out.root";
   if (!gSystem->AccessPathName( fname )) {
      input = TFile::Open( fname ); // check if file in local directory exists
   }
   else {
      //TFile::SetCacheFileDir(".");
      //input = TFile::Open("http://root.cern.ch/files/tmva_reg_example.root", "CACHEREAD"); // if not: download from ROOT server
   }
   if (!input) {
      std::cout << "ERROR: could not open data file" << std::endl;
      exit(1);
   }
   std::cout << "--- TMVARegression           : Using input file: " << input->GetName() << std::endl;

   // Register the regression tree

   TTree *regTree = (TTree*)input->Get("TreeR");

   // global event weights per tree (see below for setting event-wise weights)
   Double_t regWeight  = 1.0;

   // You can add an arbitrary number of regression trees
   dataloader->AddRegressionTree( regTree, regWeight );

   // This would set individual event weights (the variables defined in the
   // expression need to exist in the original TTree)
   dataloader->SetWeightExpression( "weight", "Regression" );

   // Apply additional cuts on the signal and background samples (can be different)
   TCut mycut = ""; // for example: TCut mycut = "abs(var1)<0.5 && abs(var2-0.5)<1";

   // tell the DataLoader to use all remaining events in the trees after training for testing:
   dataloader->PrepareTrainingAndTestTree( mycut,
                                         "nTrain_Regression=1000:nTest_Regression=0:SplitMode=Random:NormMode=NumEvents:!V" );
   //
   //     dataloader->PrepareTrainingAndTestTree( mycut,
   //            "nTrain_Regression=0:nTest_Regression=0:SplitMode=Random:NormMode=NumEvents:!V" );

   // If no numbers of events are given, half of the events in the tree are used
   // for training, and the other half for testing:
   //
   //     dataloader->PrepareTrainingAndTestTree( mycut, "SplitMode=random:!V" );

   // Book MVA methods
   //
   // Please lookup the various method configuration options in the corresponding cxx files, eg:
   // src/MethoCuts.cxx, etc, or here: http://tmva.sourceforge.net/optionRef.html
   // it is possible to preset ranges in the option string in which the cut optimisation should be done:
   // "...:CutRangeMin[2]=-1:CutRangeMax[2]=1"...", where [2] is the third input variable

   // PDE - RS method
   if (Use["PDERS"])
      factory->BookMethod( dataloader,  TMVA::Types::kPDERS, "PDERS",
                           "!H:!V:NormTree=T:VolumeRangeMode=Adaptive:KernelEstimator=Gauss:GaussSigma=0.3:NEventsMin=40:NEventsMax=60:VarTransform=None" );
   // And the options strings for the MinMax and RMS methods, respectively:
   //
   //      "!H:!V:VolumeRangeMode=MinMax:DeltaFrac=0.2:KernelEstimator=Gauss:GaussSigma=0.3" );
   //      "!H:!V:VolumeRangeMode=RMS:DeltaFrac=3:KernelEstimator=Gauss:GaussSigma=0.3" );

   if (Use["PDEFoam"])
       factory->BookMethod( dataloader,  TMVA::Types::kPDEFoam, "PDEFoam",
			    "!H:!V:MultiTargetRegression=F:TargetSelection=Mpv:TailCut=0.001:VolFrac=0.0666:nActiveCells=500:nSampl=2000:nBin=5:Compress=T:Kernel=None:Nmin=10:VarTransform=None" );

   // K-Nearest Neighbour classifier (KNN)
   if (Use["KNN"])
      factory->BookMethod( dataloader,  TMVA::Types::kKNN, "KNN",
                           "nkNN=20:ScaleFrac=0.8:SigmaFact=1.0:Kernel=Gaus:UseKernel=F:UseWeight=T:!Trim" );

   // Linear discriminant
   if (Use["LD"])
      factory->BookMethod( dataloader,  TMVA::Types::kLD, "LD",
                           "!H:!V:VarTransform=None" );

	// Function discrimination analysis (FDA) -- test of various fitters - the recommended one is Minuit (or GA or SA)
   if (Use["FDA_MC"])
      factory->BookMethod( dataloader,  TMVA::Types::kFDA, "FDA_MC",
                          "!H:!V:Formula=(0)+(1)*x0+(2)*x1:ParRanges=(-100,100);(-100,100);(-100,100):FitMethod=MC:SampleSize=100000:Sigma=0.1:VarTransform=D" );

   if (Use["FDA_GA"]) // can also use Simulated Annealing (SA) algorithm (see Cuts_SA options) .. the formula of this example is good for parabolas
      factory->BookMethod( dataloader,  TMVA::Types::kFDA, "FDA_GA",
                           "!H:!V:Formula=(0)+(1)*x0+(2)*x1:ParRanges=(-100,100);(-100,100);(-100,100):FitMethod=GA:PopSize=100:Cycles=3:Steps=30:Trim=True:SaveBestGen=1:VarTransform=Norm" );

   if (Use["FDA_MT"])
      factory->BookMethod( dataloader,  TMVA::Types::kFDA, "FDA_MT",
                           "!H:!V:Formula=(0)+(1)*x0+(2)*x1:ParRanges=(-100,100);(-100,100);(-100,100);(-10,10):FitMethod=MINUIT:ErrorLevel=1:PrintLevel=-1:FitStrategy=2:UseImprove:UseMinos:SetBatch" );

   if (Use["FDA_GAMT"])
      factory->BookMethod( dataloader,  TMVA::Types::kFDA, "FDA_GAMT",
                           "!H:!V:Formula=(0)+(1)*x0+(2)*x1:ParRanges=(-100,100);(-100,100);(-100,100):FitMethod=GA:Converger=MINUIT:ErrorLevel=1:PrintLevel=-1:FitStrategy=0:!UseImprove:!UseMinos:SetBatch:Cycles=1:PopSize=5:Steps=5:Trim" );

   // Neural network (MLP)
   if (Use["MLP"])
      factory->BookMethod( dataloader,  TMVA::Types::kMLP, "MLP", "!H:!V:VarTransform=Norm:NeuronType=tanh:NCycles=20000:HiddenLayers=N+20:TestRate=6:TrainingMethod=BFGS:Sampling=0.3:SamplingEpoch=0.8:ConvergenceImprove=1e-6:ConvergenceTests=15:!UseRegulator" );

   if (Use["DNN_CPU"]) {
      /*
          TString layoutString ("Layout=TANH|(N+100)*2,LINEAR");
          TString layoutString ("Layout=SOFTSIGN|100,SOFTSIGN|50,SOFTSIGN|20,LINEAR");
          TString layoutString ("Layout=RELU|300,RELU|100,RELU|30,RELU|10,LINEAR");
          TString layoutString ("Layout=SOFTSIGN|50,SOFTSIGN|30,SOFTSIGN|20,SOFTSIGN|10,LINEAR");
          TString layoutString ("Layout=TANH|50,TANH|30,TANH|20,TANH|10,LINEAR");
          TString layoutString ("Layout=SOFTSIGN|50,SOFTSIGN|20,LINEAR");
          TString layoutString ("Layout=TANH|100,TANH|30,LINEAR");
       */
      TString layoutString("Layout=TANH|50,Layout=TANH|50,Layout=TANH|50,LINEAR");

      TString training0("LearningRate=1e-2,Momentum=0.5,Repetitions=1,ConvergenceSteps=20,BatchSize=50,"
                        "TestRepetitions=10,WeightDecay=0.01,Regularization=NONE,DropConfig=0.2+0.2+0.2+0.,"
                        "DropRepetitions=2");
      TString training1("LearningRate=1e-3,Momentum=0.9,Repetitions=1,ConvergenceSteps=20,BatchSize=50,"
                        "TestRepetitions=5,WeightDecay=0.01,Regularization=L2,DropConfig=0.1+0.1+0.1,DropRepetitions="
                        "1");
      TString training2("LearningRate=1e-4,Momentum=0.3,Repetitions=1,ConvergenceSteps=10,BatchSize=50,"
                        "TestRepetitions=5,WeightDecay=0.01,Regularization=NONE");

      TString trainingStrategyString("TrainingStrategy=");
      trainingStrategyString += training0 + "|" + training1 + "|" + training2;

      //       TString trainingStrategyString
      //       ("TrainingStrategy=LearningRate=1e-1,Momentum=0.3,Repetitions=3,ConvergenceSteps=20,BatchSize=30,TestRepetitions=7,WeightDecay=0.0,L1=false,DropFraction=0.0,DropRepetitions=5");

      TString nnOptions(
         "!H:V:ErrorStrategy=SUMOFSQUARES:VarTransform=G:WeightInitialization=XAVIERUNIFORM:Architecture=CPU");
      //       TString nnOptions ("!H:V:VarTransform=Normalize:ErrorStrategy=CHECKGRADIENTS");
      nnOptions.Append(":");
      nnOptions.Append(layoutString);
      nnOptions.Append(":");
      nnOptions.Append(trainingStrategyString);

      factory->BookMethod(dataloader, TMVA::Types::kDNN, "DNN_CPU", nnOptions); // NN
   }


   // Support Vector Machine
   if (Use["SVM"])
      factory->BookMethod( dataloader,  TMVA::Types::kSVM, "SVM", "Gamma=0.25:Tol=0.001:VarTransform=Norm" );

   // Boosted Decision Trees
   if (Use["BDT"])
     factory->BookMethod( dataloader,  TMVA::Types::kBDT, "BDT",
                           "!H:!V:NTrees=100:MinNodeSize=1.0%:BoostType=AdaBoostR2:SeparationType=RegressionVariance:nCuts=20:PruneMethod=CostComplexity:PruneStrength=30" );

   if (Use["BDTG"])
     factory->BookMethod( dataloader,  TMVA::Types::kBDT, "BDTG",
                           "!H:!V:NTrees=20000::BoostType=Grad:Shrinkage=0.1:UseBaggedBoost:BaggedSampleFraction=0.5:nCuts=20:MaxDepth=10:MaxDepth=15" );
   // --------------------------------------------------------------------------------------------------

   // Now you can tell the factory to train, test, and evaluate the MVAs

   // Train MVAs using the set of training events
   factory->TrainAllMethods();

   // Evaluate all MVAs using the set of test events
   factory->TestAllMethods();

   // Evaluate and compare performance of all configured MVAs
   factory->EvaluateAllMethods();

   // --------------------------------------------------------------

   // Save the output
   outputFile->Close();

   std::cout << "==> Wrote root file: " << outputFile->GetName() << std::endl;
   std::cout << "==> TMVARegression is done!" << std::endl;

   delete factory;
   delete dataloader;

   // Launch the GUI for the root macros
   if (!gROOT->IsBatch()) TMVA::TMVARegGui( outfileName );
}
