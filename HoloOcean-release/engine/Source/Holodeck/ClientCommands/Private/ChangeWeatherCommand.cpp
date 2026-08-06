// MIT License (c) 2019 BYU PCCL see LICENSE file

#include "ChangeWeatherCommand.h"
#include "Holodeck.h"
#include "HolodeckGameMode.h"
#include "WeatherManager.h"

#include "Components/DirectionalLightComponent.h"
#include "Components/LightComponent.h"
#include "Components/PrimitiveComponent.h"
#include "Components/SkyAtmosphereComponent.h"
#include "Components/SkyLightComponent.h"
#include "Components/VolumetricCloudComponent.h"
#include "Engine/DirectionalLight.h"
#include "Engine/ExponentialHeightFog.h"
#include "Engine/SkyLight.h"
#include "NiagaraComponent.h"
#include "NiagaraFunctionLibrary.h"

class ASkyAtmosphere;
class AVolumetricCloud;

void UChangeWeatherCommand::Execute() {
	UE_LOG(LogHolodeck, Log, TEXT("UChangeWeatherCommand::Execute"));

	if (NumberParams.size() != 1) {
		UE_LOG(
			LogHolodeck,
			Error,
			TEXT(
				"Unexpected argument length found in UChangeWeatherCommand. "
				"Command not executed."));
		return;
	}

	AHolodeckGameMode* GameTarget = static_cast<AHolodeckGameMode*>(Target);
	if (GameTarget == nullptr) {
		UE_LOG(
			LogHolodeck,
			Warning,
			TEXT(
				"UCommand::Target is not a UHolodeckGameMode*. "
				"UChangeWeatherCommand::Weather not changed."));
		return;
	}

	UWorld* World = Target->GetWorld();
	if (World == nullptr) {
		UE_LOG(
			LogHolodeck,
			Warning,
			TEXT(
				"UChangeWeatherCommand::Execute found world as nullptr. "
				"Weather not changed."));
		return;
	}

	// Getting actors
	TArray<AActor*> DirectionalLights;
	UGameplayStatics::GetAllActorsOfClass(
		World, ADirectionalLight::StaticClass(), DirectionalLights);
	TArray<AActor*> SkyLights;
	UGameplayStatics::GetAllActorsOfClass(World, ASkyLight::StaticClass(), SkyLights);
	TArray<AActor*> HeightFogs;
	UGameplayStatics::GetAllActorsOfClass(
		World, AExponentialHeightFog::StaticClass(), HeightFogs);
	TArray<AActor*> SkyAtmospheres;
	UGameplayStatics::GetAllActorsOfClass(
		World, ASkyAtmosphere::StaticClass(), SkyAtmospheres);
	TArray<AActor*> VolumetricClouds;
	UGameplayStatics::GetAllActorsOfClass(
		World, AVolumetricCloud::StaticClass(), VolumetricClouds);
	TArray<AActor*> Planes;
	UGameplayStatics::GetAllActorsOfClass(
		World, AStaticMeshActor::StaticClass(), Planes);

	// Checking actors exist
	if (DirectionalLights.Num() == 0 || SkyLights.Num() == 0 || HeightFogs.Num() == 0
		|| SkyAtmospheres.Num() == 0 || VolumetricClouds.Num() == 0
		|| Planes.Num() == 0) {
		UE_LOG(
			LogHolodeck,
			Warning,
			TEXT("Not enough actors found in the scene. Weather not changed."));
		return;
	}

	int weather = NumberParams[0];

	// Validate input
	if (weather < 0 || weather > 2) {
		UE_LOG(
			LogHolodeck,
			Warning,
			TEXT("Invalid weather option: %d.  (0 - sunny, 1 - cloudy, 2 - rainy)."),
			weather);
		return;
	}

	// There should only be one actor of each in the world
	ADirectionalLight* DirectionalLight = Cast<ADirectionalLight>(DirectionalLights[0]);
	ASkyLight*		   SkyLight = Cast<ASkyLight>(SkyLights[0]);
	AExponentialHeightFog* HeightFog = Cast<AExponentialHeightFog>(HeightFogs[0]);
	ASkyAtmosphere*		   SkyAtmosphere = Cast<ASkyAtmosphere>(SkyAtmospheres[0]);

	// Getting the clouds from the world
	AVolumetricCloud* SunnyClouds = nullptr;
	AVolumetricCloud* CloudyClouds = nullptr;

	for (AActor* Actor : VolumetricClouds) {
		if (Actor->Tags.Contains(FName("Sunny"))) {
			SunnyClouds = Cast<AVolumetricCloud>(Actor);
			UE_LOG(LogHolodeck, Display, TEXT("Found Sunny VolumetricCloud actor."));
		} else if (Actor->Tags.Contains(FName("Cloudy"))) {
			CloudyClouds = Cast<AVolumetricCloud>(Actor);
			UE_LOG(LogHolodeck, Display, TEXT("Found Cloudy VolumetricCloud actor."));
		}
	}
	if (!SunnyClouds || !CloudyClouds) {
		UE_LOG(
			LogHolodeck,
			Warning,
			TEXT("Could not find both Sunny and Cloudy VolumetricCloud actors."));
		return;
	}

	// Getting the water planes, most worls only have 1 but there may be more
	TArray<AActor*> WaterPlanes;
	for (AActor* actor : Planes) {
		if (actor->Tags.Contains("WaterSurface")) {
			WaterPlanes.Add(actor);
		}
	}

	// Actor's components
	UDirectionalLightComponent* DirectionalLightComponent =
		Cast<UDirectionalLightComponent>(DirectionalLight->GetLightComponent());
	USkyLightComponent* SkyLightComponent =
		Cast<USkyLightComponent>(SkyLight->GetLightComponent());
	UExponentialHeightFogComponent* HeightFogComponent = HeightFog->GetComponent();
	USkyAtmosphereComponent* SkyAtmosphereComponent = Cast<USkyAtmosphereComponent>(
		SkyAtmosphere->GetComponentByClass(USkyAtmosphereComponent::StaticClass()));

	if (!DirectionalLightComponent || !SkyLightComponent || !HeightFogComponent
		|| !SkyAtmosphereComponent) {
		UE_LOG(
			LogHolodeck,
			Warning,
			TEXT("One or more weather components are null. Weather not changed."));
		return;
	}

	// Sky Atmosphere color (RGB)
	FLinearColor SunnyRayleighScattering(0.15f, 0.5f, 1.0f);
	FLinearColor CloudyRayleighScattering(0.4f, 0.5f, 1.0f);

	// Weather settings
	if (weather == 0) // Sunny
	{
		DirectionalLightComponent->SetIntensity(10);
		SkyLightComponent->SetIntensity(5);
		HeightFogComponent->SetFogDensity(0.02);
		HeightFogComponent->SetFogHeightFalloff(0.2);
		SkyAtmosphereComponent->SetRayleighScattering(SunnyRayleighScattering);
		SkyAtmosphereComponent->SetRayleighExponentialDistribution(4);
		CloudyClouds->SetActorHiddenInGame(true);
		SunnyClouds->SetActorHiddenInGame(false);

		// Find WeatherManager in the world and set Is Rainy
		for (TActorIterator<AWeatherManager> It(GameTarget->GetWorld()); It; ++It) {
			AWeatherManager* Manager = *It;
			if (Manager) {
				Manager->SetIsRainy(false);
				break;
			}
		}

		// Change Water material to normal water
		for (AActor* WaterPlane : WaterPlanes) {
			AStaticMeshActor* MeshActor = Cast<AStaticMeshActor>(WaterPlane);
			if (MeshActor) {
				UStaticMeshComponent* MeshComp = MeshActor->GetStaticMeshComponent();
				if (MeshComp) {
					FString PreviousMaterial = MeshComp->GetMaterial(0)->GetName();
					// Get new material
					if (PreviousMaterial.Contains("DamFlat")) {
						UMaterialInterface* NewMaterial =
							LoadObject<UMaterialInterface>(
								nullptr,
								TEXT(
									"/Game/Worlds/WaterMaterial/"
									"MM_Ocean_DamFlat.MM_Ocean_DamFlat"));
						// Check new material
						if (NewMaterial) {
							// Apply new material
							MeshComp->SetMaterial(0, NewMaterial);
						} else {
							UE_LOG(
								LogHolodeck, Warning, TEXT("New material not found"));
						}
					} else if (PreviousMaterial.Contains("OpenWater")) {
						UMaterialInterface* NewMaterial =
							LoadObject<UMaterialInterface>(
								nullptr,
								TEXT(
									"/Game/Worlds/WaterMaterial/"
									"MM_Ocean_OpenWater.MM_Ocean_OpenWater"));
						if (NewMaterial) {
							// Apply new material
							MeshComp->SetMaterial(0, NewMaterial);
						} else {
							UE_LOG(
								LogHolodeck, Warning, TEXT("New material not found"));
						}
					} else if (PreviousMaterial.Contains("Simple")) {
						UMaterialInterface* NewMaterial =
							LoadObject<UMaterialInterface>(
								nullptr,
								TEXT(
									"/Game/Worlds/WaterMaterial/"
									"MM_Ocean_Simple.MM_Ocean_Simple"));
						if (NewMaterial) {
							// Apply new material
							MeshComp->SetMaterial(0, NewMaterial);
						} else {
							UE_LOG(
								LogHolodeck, Warning, TEXT("New material not found"));
						}
					} else if (PreviousMaterial.Contains("Test")) {
						UMaterialInterface* NewMaterial =
							LoadObject<UMaterialInterface>(
								nullptr,
								TEXT(
									"/Game/StarterContent/Materials/"
									"M_TranslucentBlue_Water.M_TranslucentBlue_Water"));
						if (NewMaterial) {
							// Apply new material
							MeshComp->SetMaterial(0, NewMaterial);
						} else {
							UE_LOG(
								LogHolodeck, Warning, TEXT("New material not found"));
						}
					}
				}
			}
		}
	} else if (weather == 1) // Cloudy
	{
		DirectionalLightComponent->SetIntensity(1);
		SkyLightComponent->SetIntensity(1);
		HeightFogComponent->SetFogDensity(0.1);
		HeightFogComponent->SetFogHeightFalloff(0.2);
		SkyAtmosphereComponent->SetRayleighScattering(CloudyRayleighScattering);
		SkyAtmosphereComponent->SetRayleighExponentialDistribution(7);
		SunnyClouds->SetActorHiddenInGame(true);
		CloudyClouds->SetActorHiddenInGame(false);

		// Find WeatherManager in the world and set Is Rainy
		for (TActorIterator<AWeatherManager> It(GameTarget->GetWorld()); It; ++It) {
			AWeatherManager* Manager = *It;
			if (Manager) {
				Manager->SetIsRainy(false);
				break;
			}
		}

		// Change Water material to normal water
		for (AActor* WaterPlane : WaterPlanes) {
			AStaticMeshActor* MeshActor = Cast<AStaticMeshActor>(WaterPlane);
			if (MeshActor) {
				UStaticMeshComponent* MeshComp = MeshActor->GetStaticMeshComponent();
				if (MeshComp) {
					FString PreviousMaterial = MeshComp->GetMaterial(0)->GetName();
					// Get new material
					if (PreviousMaterial.Contains("DamFlat")) {
						UMaterialInterface* NewMaterial =
							LoadObject<UMaterialInterface>(
								nullptr,
								TEXT(
									"/Game/Worlds/WaterMaterial/"
									"MM_Ocean_DamFlat.MM_Ocean_DamFlat"));
						// Check new material
						if (NewMaterial) {
							// Apply new material
							MeshComp->SetMaterial(0, NewMaterial);
						} else {
							UE_LOG(
								LogHolodeck, Warning, TEXT("New material not found"));
						}
					} else if (PreviousMaterial.Contains("OpenWater")) {
						UMaterialInterface* NewMaterial =
							LoadObject<UMaterialInterface>(
								nullptr,
								TEXT(
									"/Game/Worlds/WaterMaterial/"
									"MM_Ocean_OpenWater.MM_Ocean_OpenWater"));
						if (NewMaterial) {
							// Apply new material
							MeshComp->SetMaterial(0, NewMaterial);
						} else {
							UE_LOG(
								LogHolodeck, Warning, TEXT("New material not found"));
						}
					} else if (PreviousMaterial.Contains("Simple")) {
						UMaterialInterface* NewMaterial =
							LoadObject<UMaterialInterface>(
								nullptr,
								TEXT(
									"/Game/Worlds/WaterMaterial/"
									"MM_Ocean_Simple.MM_Ocean_Simple"));
						if (NewMaterial) {
							// Apply new material
							MeshComp->SetMaterial(0, NewMaterial);
						} else {
							UE_LOG(
								LogHolodeck, Warning, TEXT("New material not found"));
						}
					} else if (PreviousMaterial.Contains("Test")) {
						UMaterialInterface* NewMaterial =
							LoadObject<UMaterialInterface>(
								nullptr,
								TEXT(
									"/Game/StarterContent/Materials/"
									"M_TranslucentBlue_Water.M_TranslucentBlue_Water"));
						if (NewMaterial) {
							// Apply new material
							MeshComp->SetMaterial(0, NewMaterial);
						} else {
							UE_LOG(
								LogHolodeck, Warning, TEXT("New material not found"));
						}
					}
				}
			}
		}
	} else if (weather == 2) // Rainy
	{
		// UE_LOG(LogHolodeck, Warning, TEXT("Entered rainy."));
		DirectionalLightComponent->SetIntensity(1);
		SkyLightComponent->SetIntensity(1);
		HeightFogComponent->SetFogDensity(0.1);
		HeightFogComponent->SetFogHeightFalloff(0.2);
		SkyAtmosphereComponent->SetRayleighScattering(CloudyRayleighScattering);
		SkyAtmosphereComponent->SetRayleighExponentialDistribution(7);
		SunnyClouds->SetActorHiddenInGame(true);
		CloudyClouds->SetActorHiddenInGame(false);

		// Find WeatherManager in the world and set Is Rainy
		for (TActorIterator<AWeatherManager> It(GameTarget->GetWorld()); It; ++It) {
			AWeatherManager* Manager = *It;
			if (Manager) {
				Manager->SetIsRainy(true);
				break;
			}
		}

		// Change Water material to create ripples
		for (AActor* WaterPlane : WaterPlanes) {
			AStaticMeshActor* MeshActor = Cast<AStaticMeshActor>(WaterPlane);
			if (MeshActor) {
				UStaticMeshComponent* MeshComp = MeshActor->GetStaticMeshComponent();
				if (MeshComp) {
					FString PreviousMaterial = MeshComp->GetMaterial(0)->GetName();
					// Get new material
					if (PreviousMaterial.Contains("DamFlat")) {
						UMaterialInterface* NewMaterial = LoadObject<
							UMaterialInterface>(
							nullptr,
							TEXT(
								"/Game/WeatherContent/WaterRipples/"
								"MM_Ocean_Ripples_DamFlat.MM_Ocean_Ripples_DamFlat"));
						// Check new material
						if (NewMaterial) {
							// Apply new material
							MeshComp->SetMaterial(0, NewMaterial);
						} else {
							UE_LOG(
								LogHolodeck, Warning, TEXT("New material not found"));
						}
					} else if (PreviousMaterial.Contains("OpenWater")) {
						UMaterialInterface* NewMaterial = LoadObject<
							UMaterialInterface>(
							nullptr,
							TEXT(
								"/Game/WeatherContent/WaterRipples/"
								"MM_Ocean_Ripples_OpenWater.MM_Ocean_Ripples_OpenWater"));
						if (NewMaterial) {
							// Apply new material
							MeshComp->SetMaterial(0, NewMaterial);
						} else {
							UE_LOG(
								LogHolodeck, Warning, TEXT("New material not found"));
						}
					} else if (PreviousMaterial.Contains("Simple")) {
						UMaterialInterface* NewMaterial =
							LoadObject<UMaterialInterface>(
								nullptr,
								TEXT(
									"/Game/WeatherContent/WaterRipples/"
									"MM_Ocean_Ripples_Simple.MM_Ocean_Ripples_Simple"));
						if (NewMaterial) {
							// Apply new material
							MeshComp->SetMaterial(0, NewMaterial);
						} else {
							UE_LOG(
								LogHolodeck, Warning, TEXT("New material not found"));
						}
					} else if (PreviousMaterial.Contains("TranslucentBlue")) {
						UMaterialInterface* NewMaterial =
							LoadObject<UMaterialInterface>(
								nullptr,
								TEXT(
									"/Game/WeatherContent/WaterRipples/"
									"MM_Ocean_Ripples_Test.MM_Ocean_Ripples_Test"));
						if (NewMaterial) {
							// Apply new material
							MeshComp->SetMaterial(0, NewMaterial);
						} else {
							UE_LOG(
								LogHolodeck, Warning, TEXT("New material not found"));
						}
					}
				}
			}
		}
	}
}