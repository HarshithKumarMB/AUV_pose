#pragma once
#include "Command.h"
#include "Holodeck.h"

#include "WaterFogCommand.generated.h"

/**
 * WaterFogCommand
 * Command used to change the Water Height Fog
 * NumberParams expects 5 parameters: fog density, fog depth, and fog color
 * (R,G,B).
 */

UCLASS()
class HOLODECK_API UWaterFogCommand : public UCommand {
	GENERATED_BODY()
public:
	// See UCommand for the documentation of this overridden function.
	void Execute() override;
};