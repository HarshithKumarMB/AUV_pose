#pragma once

#include "Holodeck.h"

#include "Command.h"

#include "WaterColorCommand.generated.h"

/**
 * Command to change the color of the water
 *
 *
 * NumberParams expects two arguments representing either the amount to change
 * the surface level by and a 0 or the new surface level and a 1
 */
UCLASS()
class HOLODECK_API UWaterColorCommand : public UCommand {
	GENERATED_BODY()

public:
	// See UCommand for the documentation of this overridden function.
	void Execute() override;
};
