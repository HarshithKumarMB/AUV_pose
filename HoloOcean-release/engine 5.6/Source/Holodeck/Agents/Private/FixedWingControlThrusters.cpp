#include "FixedWingControlThrusters.h"
#include "Holodeck.h"

UFixedWingControlThrusters::UFixedWingControlThrusters(
	const FObjectInitializer& ObjectInitializer)
	: Super(ObjectInitializer) {}

void UFixedWingControlThrusters::Execute(
	void* const CommandArray,
	void* const InputCommand,
	float		DeltaSeconds) {
	if (FixedWing == nullptr) {
		FixedWing = static_cast<AFixedWing*>(FixedWingController->GetPawn());
		if (FixedWing == nullptr) {
			UE_LOG(
				LogHolodeck,
				Error,
				TEXT("UFixedWingControlThrusters couldn't get FixedWing reference"));
			return;
		}

		FixedWing->EnableDamping();
	}

	float* InputCommandFloat = static_cast<float*>(InputCommand);
	float* CommandArrayFloat = static_cast<float*>(CommandArray);

	FixedWing->ApplyBuoyancyDragForce();
	FixedWing->ApplyThrusters(InputCommandFloat);

	// Zero out the physics based controller
	for (int i = 0; i < 6; i++) {
		CommandArrayFloat[i] = 0;
	}
}