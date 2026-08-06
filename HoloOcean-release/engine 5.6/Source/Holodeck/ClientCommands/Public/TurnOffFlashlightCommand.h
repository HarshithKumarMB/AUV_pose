#pragma once

#include "Command.h"
#include "Holodeck.h"

#include "TurnOffFlashlightCommand.generated.h"

/**
 * TurnOffFlashlightCommand
 * Command used to turn off the vehicle's flashlight
 * NumberParams expects 1 argument, Flashlight name
 */

UCLASS()
class HOLODECK_API UTurnOffFlashlightCommand : public UCommand {
	GENERATED_BODY()

public:
	// See UCommand for the documentation of this overridden function.
	void Execute() override;
};