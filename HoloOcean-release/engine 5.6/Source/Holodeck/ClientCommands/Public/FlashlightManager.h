#pragma once

#include "GameFramework/Actor.h"

#include "FlashlightManager.generated.h"

UCLASS()
class HOLODECK_API AFlashlightManager : public AActor {
	GENERATED_BODY()

private:
	UPROPERTY(
		EditAnywhere,
		BlueprintReadWrite,
		Category = "Flashlight",
		meta = (AllowPrivateAccess = "true"))
	int lightNumber;

	UPROPERTY(
		EditAnywhere,
		BlueprintReadWrite,
		Category = "Flashlight",
		meta = (AllowPrivateAccess = "true"))
	float lightIntensity;

	UPROPERTY(
		EditAnywhere,
		BlueprintReadWrite,
		Category = "Flashlight",
		meta = (AllowPrivateAccess = "true"))
	float beamWidth;

	UPROPERTY(
		EditAnywhere,
		BlueprintReadWrite,
		Category = "Flashlight",
		meta = (AllowPrivateAccess = "true"))
	FRotator lightAngle;

	UPROPERTY(
		EditAnywhere,
		BlueprintReadWrite,
		Category = "Flashlight",
		meta = (AllowPrivateAccess = "true"))
	FVector lightPosition;

	UPROPERTY(
		EditAnywhere,
		BlueprintReadWrite,
		Category = "Flashlight",
		meta = (AllowPrivateAccess = "true"))
	FLinearColor lightColor;

public:
	AFlashlightManager();

	UFUNCTION(BlueprintCallable, Category = "Flashlight")
	int GetLightNumber() const { return lightNumber; }

	UFUNCTION(BlueprintCallable, Category = "Flashlight")
	void SetLightNumber(int InLightNumber);

	UFUNCTION(BlueprintCallable, Category = "Flashlight")
	float GetLightIntensity() const { return lightIntensity; }

	UFUNCTION(BlueprintCallable, Category = "Flashlight")
	void SetLightIntensity(float InLightIntensity);

	UFUNCTION(BlueprintCallable, Category = "Flashlight")
	float GetBeamWidth() const { return beamWidth; }

	UFUNCTION(BlueprintCallable, Category = "Flashlight")
	void SetBeamWidth(float InBeamWidth);

	UFUNCTION(BlueprintCallable, Category = "Flashlight")
	FRotator GetLightAngle() const { return lightAngle; }

	UFUNCTION(BlueprintCallable, Category = "Flashlight")
	void SetLightAngle(FRotator InLightAngle);

	UFUNCTION(BlueprintCallable, Category = "Flashlight")
	FVector GetLightPosition() const { return lightPosition; }

	UFUNCTION(BlueprintCallable, Category = "Flashlight")
	void SetLightPosition(FVector InLightPosition);

	UFUNCTION(BlueprintCallable, Category = "Flashlight")
	FLinearColor GetLightColor() const { return lightColor; }

	UFUNCTION(BlueprintCallable, Category = "Flashlight")
	void SetLightColor(FLinearColor InLightColor);

	UFUNCTION(BlueprintImplementableEvent, Category = "Flashlight")
	void TriggerToggleFlashlight();
};