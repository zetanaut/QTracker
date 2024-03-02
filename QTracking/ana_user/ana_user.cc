//_____________________________________________________________________________
// Standard Headers:
#include <fstream>
#include <cmath>
#include <iostream>
#include <unistd.h>
#include <vector>
#include "TFile.h"
#include "TTree.h"
#include "TH1.h"
#include "TH2F.h"
#include "TF1.h"
#include "TMath.h"
#include "TString.h"
#include "TSystem.h"
#include "TFile.h"
#include "TROOT.h"
#include <TVector3.h>
#include "TLorentzVector.h"
#include "TApplication.h"
#include <string.h>
#include "TRandom.h"

using namespace std;

int main(int __argc,char *__argv[]){

   Int_t Argc = __argc;
   char **Input = __argv;
   std::vector<string> w;
   w.assign(__argv, __argv + __argc);  
   TApplication* theApp = new TApplication("App", &__argc, __argv);   
   char *outFileName = (char *) "Data_Trees.root";
   extern int optind;   
   
   TFile outFile(outFileName,"recreate");  
   Float_t Gpx, Gpy, Gpz, px, py, pz;

   
   TH1F *h1 = new TH1F("dx", "dx", 200, -3, 3);
   TH1F *h2 = new TH1F("dy", "dy", 200, -3, 3);
   TH1F *h3 = new TH1F("dz", "dz", 200, -10, 10);   
    
   
  
   for(int n_arg = optind; n_arg < Argc; n_arg++){
  	TString input =w[n_arg]; 
   	TFile inFile(input); // open the input file     

   if(TTree *TreeFin = (TTree*)inFile.Get("TreeFin")){
            
   TreeFin->SetBranchAddress("Gpx",&Gpx);
   TreeFin->SetBranchAddress("Gpy",&Gpy);   
   TreeFin->SetBranchAddress("Gpz",&Gpz);   
   TreeFin->SetBranchAddress("px",&px);
   TreeFin->SetBranchAddress("py",&py);   
   TreeFin->SetBranchAddress("pz",&pz);     
 
   Int_t nentries = (Int_t)TreeFin->GetEntries();
 
   for (Int_t j=0;j<=nentries;j++) { TreeFin->GetEntry(j);
  
   h1->Fill(Gpx-px);
   h2->Fill(Gpy-py);   
   h3->Fill(Gpz-pz);   
 
      
}
}
  	else {
            cout << "File has no TTree " << endl;
        }  

  }//n_arg  

   outFile.Write(); // write to the output file
   outFile.Close(); // close the output file
       

}//end of main
