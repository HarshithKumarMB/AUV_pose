#pragma once

#include "Command.h"
#include "Holodeck.h"

#include "ChangeWeatherCommand.generated.h"

/**
 * ChangeWeatherCommand
 * Command used to change the weather in the world
 * NumberParams expects 1 argument representing the weather:
 * 0 - sunny, 1 - cloudy, 2 - rainy
 */

UCLASS()
class HOLODECK_API UChangeWeatherCommand : public UCommand {
	GENERATED_BODY()

public:
	// See UCommand for the documentation of this overridden function.
	void Execute() override;
};
