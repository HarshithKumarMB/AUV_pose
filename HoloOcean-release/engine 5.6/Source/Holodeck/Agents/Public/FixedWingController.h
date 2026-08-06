// MIT License (c) 2021 BYU FRoStLab see LICENSE file

#pragma once

#include "Holodeck.h"

#include "FixedWing.h"
#include "FixedWingControlPD.h"
#include "FixedWingControlThrusters.h"
#include "HolodeckPawnController.h"

#include "FixedWingController.generated.h"

/**
 * A Holodeck Turtle Agent Controller
 */
UCLASS()
class HOLODECK_API AFixedWingController : public AHolodeckPawnController {
	GENERATED_BODY()

public:
	/**
	 * Default Constructor
	 */
	AFixedWingController(
		const FObjectInitializer& ObjectInitializer = FObjectInitializer::Get());

	/**
	 * Default Destructor
	 */
	~AFixedWingController();

	void AddControlSchemes() override {
		// The default controller currently in ControlSchemes index 0 is the
		// dynamics one. We push it back to index 2 with this code.

		// Thruster controller
		UFixedWingControlThrusters* Thrusters = NewObject<UFixedWingControlThrusters>();
		Thrusters->SetController(this);
		ControlSchemes.Insert(Thrusters, 0);

		// Position / orientation controller
		UFixedWingControlPD* PD = NewObject<UFixedWingControlPD>();
		PD->SetController(this);
		ControlSchemes.Insert(PD, 1);
	}
};
