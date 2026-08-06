#pragma once

#include "Holodeck.h"
#include "BlueROV2.h"
#include "HolodeckControlScheme.h"
#include "HolodeckPawnController.h"
#include "SimplePID.h"
#include <math.h>

#include "BlueROV2ControlPD.generated.h"

const float BR_CONTROL_MAX_LIN_ACCEL = 1;
const float BR_CONTROL_MAX_ANG_ACCEL = 1;

const float BR_POS_X_P = 100;
const float BR_POS_X_I = 0;
const float BR_POS_X_D = 50;

const float BR_POS_Y_P = 100;
const float BR_POS_Y_I = 0;
const float BR_POS_Y_D = 50;

const float BR_POS_Z_P = 100;
const float BR_POS_Z_I = 0;
const float BR_POS_Z_D = 50;

const float BR_ROT_R_P = 0.2;
const float BR_ROT_R_I = 0.001;
const float BR_ROT_R_D = 0.15;

const float BR_ROT_P_P = 0.2;
const float BR_ROT_P_I = 0.001;
const float BR_ROT_P_D = 0.15;

const float BR_ROT_Y_P = 0.2;
const float BR_ROT_Y_I = 0.001;
const float BR_ROT_Y_D = 0.15;
/**
 * BlueROV2ControlPD
 */
UCLASS()
class HOLODECK_API UBlueROV2ControlPD : public UHolodeckControlScheme {
public:
	GENERATED_BODY()

	UBlueROV2ControlPD(const FObjectInitializer& ObjectInitializer);

	void Execute(void* const CommandArray, void* const InputCommand, float DeltaSeconds)
		override;

	unsigned int GetControlSchemeSizeInBytes() const override {
		return 8 * sizeof(float);
	}

	void SetController(AHolodeckPawnController* const Controller) {
		BlueROV2Controller = Controller;
	};

private:
	AHolodeckPawnController* BlueROV2Controller;
	ABlueROV2*				 BlueROV2;
	SimplePID				 PositionController_X;
	SimplePID				 PositionController_Y;
	SimplePID				 PositionController_Z;
	SimplePID				 RotationController_R;
	SimplePID				 RotationController_P;
	SimplePID				 RotationController_Y;
};
