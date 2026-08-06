#pragma once

#include "Holodeck.h"
#include "CougUV.h"
#include "HolodeckControlScheme.h"
#include "HolodeckPawnController.h"
#include <math.h>

#include "CougUVControlFins.generated.h"

/**
 * UCougUVControlFins
 */
UCLASS()
class HOLODECK_API UCougUVControlFins : public UHolodeckControlScheme {
public:
	GENERATED_BODY()

	UCougUVControlFins(const FObjectInitializer& ObjectInitializer);

	void Execute(void* const CommandArray, void* const InputCommand, float DeltaSeconds)
		override;

	/** NOTE: These go counter-clockwise, starting in front right
	 * 0: Left Fin
	 * 1: Top Fin
	 * 2: Right Fin
	 * 3: Bottom Fin
	 * 4: Thruster
	 */
	unsigned int GetControlSchemeSizeInBytes() const override {
		return 5 * sizeof(float);
	}

	void SetController(AHolodeckPawnController* const Controller) {
		CougUVController = Controller;
	};

private:
	AHolodeckPawnController* CougUVController;
	ACougUV*				 CougUV;
};
