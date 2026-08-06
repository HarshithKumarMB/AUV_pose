#include "CougUVControlFins.h"
#include "Holodeck.h"

UCougUVControlFins::UCougUVControlFins(const FObjectInitializer& ObjectInitializer)
	: Super(ObjectInitializer) {}

void UCougUVControlFins::Execute(
	void* const CommandArray,
	void* const InputCommand,
	float		DeltaSeconds) {
	if (CougUV == nullptr) {
		CougUV = static_cast<ACougUV*>(CougUVController->GetPawn());
		if (CougUV == nullptr) {
			UE_LOG(
				LogHolodeck,
				Error,
				TEXT("UCougUVControlFins couldn't get CougUV reference"));
			return;
		}

		CougUV->EnableDamping();
	}

	float* InputCommandFloat = static_cast<float*>(InputCommand);
	float* CommandArrayFloat = static_cast<float*>(CommandArray);

	// Buoyancy & Drag forces
	CougUV->ApplyBuoyancyDragForce();

	// Propeller
	CougUV->ApplyThrust(InputCommandFloat[4]);

	// Apply fin forces
	for (int i = 0; i < 4; i++) {
		CougUV->ApplyFin(i, InputCommandFloat[i]);
	}

	// Zero out the physics based controller
	for (int i = 0; i < 6; i++) {
		CommandArrayFloat[i] = 0;
	}
}