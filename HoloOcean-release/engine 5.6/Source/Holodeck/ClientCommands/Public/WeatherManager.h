#pragma once

#include "GameFramework/Actor.h"

#include "WeatherManager.generated.h"

UCLASS()
class HOLODECK_API AWeatherManager : public AActor {
	GENERATED_BODY()

private:
	UPROPERTY(
		VisibleAnywhere,
		BlueprintReadOnly,
		Category = "Weather",
		meta = (AllowPrivateAccess = "true"))
	bool bIsRainy;

	UPROPERTY(
		EditAnywhere,
		BlueprintReadWrite,
		Category = "Weather",
		meta = (AllowPrivateAccess = "true"))
	FVector rainVelocity;

	UPROPERTY(
		EditAnywhere,
		BlueprintReadWrite,
		Category = "Weather",
		meta = (AllowPrivateAccess = "true"))
	float spawnRate;

public:
	AWeatherManager();

	UFUNCTION(BlueprintCallable, Category = "Weather")
	bool GetIsRainy() const { return bIsRainy; }

	UFUNCTION(BlueprintCallable, Category = "Weather")
	void SetIsRainy(bool bRain);

	UFUNCTION(BlueprintCallable, Category = "Weather")
	FVector GetRainVelocity() const { return rainVelocity; }

	UFUNCTION(BlueprintCallable, Category = "Weather")
	void SetRainVelocity(FVector newRainVelocity);

	UFUNCTION(BlueprintCallable, Category = "Weather")
	float GetSpawnRate() const { return spawnRate; }

	UFUNCTION(BlueprintCallable, Category = "Weather")
	void SetSpawnRate(float newSpawnRate);
};
