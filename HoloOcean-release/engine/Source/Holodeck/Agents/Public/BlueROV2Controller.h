// MIT License (c) 2021 BYU FRoStLab see LICENSE file

#pragma once

#include "Holodeck.h"
#include "BlueROV2.h"
#include "BlueROV2ControlPD.h"
#include "BlueROV2ControlThrusters.h"
#include "HolodeckPawnController.h"

#include "BlueROV2Controller.generated.h"

/**
 * A Holodeck Turtle Agent Controller
 */
UCLASS()

class HOLODECK_API ABlueROV2Controller : public AHolodeckPawnController {
	GENERATED_BODY()

public:
	/**
	 * Default Constructor
	 */
	ABlueROV2Controller(
		const FObjectInitializer& ObjectInitializer = FObjectInitializer::Get());

	/**
	 * Default Destructor
	 */
	~ABlueROV2Controller();

	void AddControlSchemes() override {
		// The default controller currently in ControlSchemes index 0 is the
		// dynamics one. We push it back to index 2 with this code.

		// Thruster controller

		UBlueROV2ControlThrusters* Thrusters = NewObject<UBlueROV2ControlThrusters>();
		Thrusters->SetController(this);
		ControlSchemes.Insert(Thrusters, 0);

		// Position / orientation controller
		UBlueROV2ControlPD* PD = NewObject<UBlueROV2ControlPD>();
		PD->SetController(this);
		ControlSchemes.Insert(PD, 1);
	}
};
