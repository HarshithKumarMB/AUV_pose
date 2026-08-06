#pragma once

#include "Command.h"
#include "Holodeck.h"

#include "AirFogCommand.generated.h"

/**
 * AirFogCommand
 * Command used to change the Air Height Fog
 * NumberParams expects 5 parameters: fog density, fog depth, and fog color
 * (R,G,B).
 */

UCLASS()
class HOLODECK_API UAirFogCommand : public UCommand {
	GENERATED_BODY()
public:
	// See UCommand for the documentation of this overridden function.
	void Execute() override;
};