#pragma once

#include "Holodeck.h"

#include "HolodeckControlScheme.h"
#include "HolodeckPawnController.h"
#include "HoveringAUV.h"
#include "SimplePID.h"
#include <math.h>

#include "HoveringAUVControlPD.generated.h"

const float AUV_CONTROL_MAX_LIN_ACCEL = 1;
const float AUV_CONTROL_MAX_ANG_ACCEL = 1;

const float AUV_POS_X_P = 100;
const float AUV_POS_X_I = 0;
const float AUV_POS_X_D = 50;

const float AUV_POS_Y_P = 100;
const float AUV_POS_Y_I = 0;
const float AUV_POS_Y_D = 50;

const float AUV_POS_Z_P = 100;
const float AUV_POS_Z_I = 0;
const float AUV_POS_Z_D = 50;

const float AUV_ROT_R_P = 0.2;
const float AUV_ROT_R_I = 0.001;
const float AUV_ROT_R_D = 0.15;

const float AUV_ROT_P_P = 0.2;
const float AUV_ROT_P_I = 0.001;
const float AUV_ROT_P_D = 0.15;

const float AUV_ROT_Y_P = 0.2;
const float AUV_ROT_Y_I = 0.001;
const float AUV_ROT_Y_D = 0.15;

/**
 * UHoveringAUVControlPD
 */
UCLASS()
class HOLODECK_API UHoveringAUVControlPD : public UHolodeckControlScheme {
public:
	GENERATED_BODY()

	UHoveringAUVControlPD(const FObjectInitializer& ObjectInitializer);

	void Execute(void* const CommandArray, void* const InputCommand, float DeltaSeconds)
		override;

	unsigned int GetControlSchemeSizeInBytes() const override {
		return 6 * sizeof(float);
	}

	void SetController(AHolodeckPawnController* const Controller) {
		HoveringAUVController = Controller;
	};

private:
	AHolodeckPawnController* HoveringAUVController;
	AHoveringAUV*			 HoveringAUV;

	SimplePID PositionController_X;
	SimplePID PositionController_Y;
	SimplePID PositionController_Z;
	SimplePID RotationController_R;
	SimplePID RotationController_P;
	SimplePID RotationController_Y;
};
