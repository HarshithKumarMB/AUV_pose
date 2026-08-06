#pragma once

#include "Command.h"
#include "Holodeck.h"

#include "TurnOnFlashlightCommand.generated.h"

/**
* TurnOnFlashlightCommand
* Command used to turn on the vehicle's flashlight and set its parameters
* NumberParams expects 11 arguments to turn the flashlight on:
* 	- Flashlight name, Light intensity, Beam width, Light location (x),
Light location (y), Light location (z), Light angle (Pitch), Light angle (Yaw),
Light color (R Component), Light color (G Component), Light color (B Component).
*/

UCLASS()
class HOLODECK_API UTurnOnFlashlightCommand : public UCommand {
	GENERATED_BODY()

public:
	// See UCommand for the documentation of this overridden function.
	void Execute() override;
};