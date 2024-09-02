#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <iostream>
#include <Eigen/Dense>

#include "GPGOMEA/Evolution/EvolutionRun.h"
#include "GPGOMEA/Evolution/EvolutionState.h"
#include "GPGOMEA/Evolution/MOArchive.h"
#include "GPGOMEA/RunHandling/IMSHandler.h"

namespace py = pybind11; 
using std::cout;
using std::endl;
using std::string;
using std::vector;

typedef Eigen::ArrayXXf MatRef;
typedef Eigen::ArrayXf VecRef;
//typedef Eigen::Ref<Eigen::ArrayXXf> MatRef;
//typedef Eigen::Ref<Eigen::ArrayXf> VecRef;

arma::mat MatRefToArma(MatRef m){
	arma::mat out(m.rows(), m.cols());
	for(int i = 0; i < m.rows(); i++)
		for(int j = 0; j < m.cols(); j++)
			out(i,j) = m(i,j);
	return out;
}

arma::vec VecRefToArma(VecRef v){
	arma::vec out(v.rows());
	for(int i = 0; i < v.rows(); i++)
		out(i) = v(i);
	return out;
}

py::list evolve(MatRef Xtrain, VecRef ytrain, MatRef Xtest = MatRef(), VecRef ytest = VecRef(), string file = "", const py::kwargs &kwargs = py::dict()) {
	// 1. SETUP
	EvolutionState * st = new EvolutionState();
	st->config->running_from_python = true;
	
	if( file.size() > 0 )  
	{
		st->SetOptionsFromFile(file);
		if( kwargs )
			std::cout << "Warning: Reading parameter settings from file. Input parameters are ignored." << std::endl;
	}
	else if( kwargs )
	{
		// Convert to cmd-like arguments
		vector<string> argv_vec;
		argv_vec.push_back("mmmogpg");
		for( auto item: kwargs )
		{
			string arg_str = "--";
			arg_str += std::string(py::str(item.first));
			string val = std::string(py::str(item.second));
			if(val.find("False") != string::npos) continue; // Skip False arguments
			else if(val.find("True") == string::npos) // For True arguments, don't add a value
			{
				arg_str += "=";
				arg_str += val;
			}
			argv_vec.push_back(arg_str);
		}
		
		// Convert to C-style array
		int argc = argv_vec.size();
		char * argv[argc];
		for(int i = 0; i < argv_vec.size(); i++){
			argv[i] = (char*) argv_vec[i].c_str();
		}
		
		std::cout << std::endl;
		st->SetOptions(argc, argv);
	}

	arma::mat aXtrain = MatRefToArma(Xtrain);
	arma::mat aytrain = VecRefToArma(ytrain);
	st->SetDataSetTraining(aXtrain, aytrain);
	if (Xtest.size() > 0 && ytest.size() > 0)
	{
		arma::mat aXtest = MatRefToArma(Xtest);
		arma::mat aytest = VecRefToArma(ytest);
		st->SetDataSetTest(aXtest, aytest);
	}
	else
	{
		arma::mat aXtest = MatRefToArma(Xtrain);
		arma::mat aytest = VecRefToArma(ytrain);
		st->SetDataSetTest(aXtest, aytest);
	}

	IMSHandler * imsh = new IMSHandler(st);

	// 2. RUN
	imsh->Start();

	// 3. OUTPUT
    MOArchive out_archive(st->fitness);
	for (EvolutionRun * r : imsh->runs) {
		if(r){
            for( auto ind : r->mo_archive.mo_archive ){
                out_archive.UpdateMOArchive(ind);
            }
        }
	}
	if (out_archive.mo_archive.empty()) {
		throw std::runtime_error("No models found, something went wrong");
	}

	py::list models;
	for (auto sol : out_archive.mo_archive ){
		string model_repr = sol->GetPythonExpression();
		models.append(model_repr);
	}

	// 4. CLEANUP
	delete imsh;

	return models;
}

PYBIND11_MODULE(_pb_mmmogpg, m) {
  m.doc() = "pybind11-based interface for MM-MO-GOMEA"; // optional module docstring
  m.def("evolve", &evolve, "Runs MM-MO-GOMEA evolution in C++",
  	py::arg("Xtrain").none(true) = py::none(), py::arg("ytrain").none(true) = py::none(), py::arg_v("Xtest", MatRef(), "None"), py::arg_v("ytest", VecRef(), "None"), py::arg("file") = "" );
}
