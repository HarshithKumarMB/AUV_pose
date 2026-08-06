#pragma once

#include "Holodeck.h"

#include "FixedWing.h"
#include "HolodeckControlScheme.h"
#include "HolodeckPawnController.h"
#include "SimplePID.h"
#include <math.h>

#include "FixedWingControlThrusters.generated.h"

/**
 * UFixedWingControlThrusters
 */
UCLASS()
class HOLODECK_API UFixedWingControlThrusters : public UHolodeckControlScheme {
public:
	GENERATED_BODY()

	UFixedWingControlThrusters(const FObjectInitializer& ObjectInitializer);

	void Execute(void* const CommandArray, void* const InputCommand, float DeltaSeconds)
		override;

	/** NOTE: These go counter-clockwise, starting in front right
	 * 0: Vertical Front Starboard Thruster
	 * 1: Vertical Front Port Thruster
	 * 2: Vertical Back  Port Thruster
	 * 3: Vertical Back  Starboard Thruster
	 * 4: Angled   Front Starboard Thruster
	 * 5: Angled   Front Port Thruster
	 * 6: Angled   Back  Port Thruster
	 * 7: Angled   Back  Starboard Thruster
	 */
	unsigned int GetControlSchemeSizeInBytes() const override {
		return 8 * sizeof(float);
	}

	void SetController(AHolodeckPawnController* const Controller) {
		FixedWingController = Controller;
	};

private:
	AHolodeckPawnController* FixedWingController;
	AFixedWing*				 FixedWing;
};
