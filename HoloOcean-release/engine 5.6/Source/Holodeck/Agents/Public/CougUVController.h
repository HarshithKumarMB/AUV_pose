// MIT License (c) 2019 BYU PCCL see LICENSE file

#pragma once

#include "Holodeck.h"
#include "CougUV.h"
#include "CougUVControlFins.h"

#include "CougUVController.generated.h"

/**
 * A Holodeck Turtle Agent Controller
 */
UCLASS()
class HOLODECK_API ACougUVController : public AHolodeckPawnController {
	GENERATED_BODY()

public:
	/**
	 * Default Constructor
	 */
	ACougUVController(
		const FObjectInitializer& ObjectInitializer = FObjectInitializer::Get());

	/**
	 * Default Destructor
	 */
	~ACougUVController();

	void AddControlSchemes() override {
		// The default controller currently in ControlSchemes index 0 is the
		// dynamics one. We push it back to index 1 with this code.

		// Thruster controller
		UCougUVControlFins* Thrusters = NewObject<UCougUVControlFins>();
		Thrusters->SetController(this);
		ControlSchemes.Insert(Thrusters, 0);
	}
};
