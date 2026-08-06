#pragma once

#include "Command.h"
#include "Holodeck.h"

#include "SetRainParametersCommand.generated.h"

/**
 * SetRainParametersCommand
 * Command used to change the rain's velocity and spawn rate
 * NumberParams expects 4 argument representing the velocity vector (v_x, v_y,
 * v_z) and spawn rate
 */

UCLASS()
class HOLODECK_API USetRainParametersCommand : public UCommand {
	GENERATED_BODY()

public:
	// See UCommand for the documentation of this overridden function.
	void Execute() override;
};
