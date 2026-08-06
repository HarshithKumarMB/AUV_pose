#include "BlueROV2ControlThrusters.h"
#include "Holodeck.h"

UBlueROV2ControlThrusters::UBlueROV2ControlThrusters(
	const FObjectInitializer& ObjectInitializer)
	: Super(ObjectInitializer) {}

void UBlueROV2ControlThrusters::Execute(
	void* const CommandArray,
	void* const InputCommand,
	float		DeltaSeconds) {
	if (BlueROV2 == nullptr) {
		BlueROV2 = static_cast<ABlueROV2*>(BlueROV2Controller->GetPawn());
		if (BlueROV2 == nullptr) {

			UE_LOG(
				LogHolodeck,
				Error,
				TEXT("UBlueROV2ControlThrusters couldn't get BlueROV2 reference"));
			return;
		}

		BlueROV2->EnableDamping();
	}

	float* InputCommandFloat = static_cast<float*>(InputCommand);
	float* CommandArrayFloat = static_cast<float*>(CommandArray);

	BlueROV2->ApplyBuoyancyDragForce();
	BlueROV2->ApplyThrusters(InputCommandFloat);

	// Zero out the physics based controller
	for (int i = 0; i < 6; i++) {
		CommandArrayFloat[i] = 0;
	}
}
