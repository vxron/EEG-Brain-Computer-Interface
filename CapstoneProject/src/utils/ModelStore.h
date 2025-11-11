#include <memory> // shared_ptr
#include "../utils/Types.h"
#include <vector>

class PatientModel_C {
public:

	// constructor requires modelId, weightMatrix and biases
	// dk if passing a pointer to weight/bias matrix is best or if i should be passing actual object by reference
	PatientModel_C(uint32_t modelId, float* weightMatrix, float* biasVec);

	PatientModel_C(const PatientModel_C&) = default;
	PatientModel_C& operator=(const PatientModel_C&) = default;
	PatientModel_C(PatientModel_C&&) = default;
	PatientModel_C& operator=(PatientModel_C&&) = default;
	~PatientModel_C() = default;

	// load model function
	std::unique_ptr<const PatientModel_C> loadModel(uint32_t modelId); // unique_ptr is owned by a single object (this one)
	uint32_t model_predict(); // output of LDA type (transform) function?
	SSVEPState_E prediction_to_ssvep_state(); // nonlinear decision function
private:
	uint32_t modelId_ = 0;
	std::vector<std::vector<float>> weightMatrix_; // 2d vector for weights
	std::vector<float> biases_;

}; // PatientModel_C