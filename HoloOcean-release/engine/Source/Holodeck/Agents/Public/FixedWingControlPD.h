#pragma once

#include "Holodeck.h"

#include "FixedWing.h"
#include "HolodeckControlScheme.h"
#include "HolodeckPawnController.h"
#include "SimplePID.h"
#include <math.h>

#include "FixedWingControlPD.generated.h"

const float FW_CONTROL_MAX_LIN_ACCEL = 1;
const float FW_CONTROL_MAX_ANG_ACCEL = 1;

const float FW_POS_P = 100;
const float FW_POS_D = 50;

const float FW_ROT_P = 0.1;
const float FW_ROT_D = 0.1;

/**
 * UFixedWingControlPD
 */
UCLASS()
class HOLODECK_API UFixedWingControlPD : public UHolodeckControlScheme {
public:
	GENERATED_BODY()

	UFixedWingControlPD(const FObjectInitializer& ObjectInitializer);

	void Execute(void* const CommandArray, void* const InputCommand, float DeltaSeconds)
		override;

	unsigned int GetControlSchemeSizeInBytes() const override {
		return 6 * sizeof(float);
	}

	void SetController(AHolodeckPawnController* const Controller) {
		FixedWingController = Controller;
	};

private:
	AHolodeckPawnController* FixedWingController;
	AFixedWing*				 FixedWing;

	SimplePID PositionController;
	SimplePID RotationController;
};
