// Copyright (c) 2020 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#pragma once

#include "LidarData.h"

#include <cstdint>
#include <vector>

namespace holoocean {
	namespace data {
		class SemanticLidarDetection : public LidarDetection {
		public:
			uint32_t object_idx{};
			uint32_t object_tag{};

			SemanticLidarDetection() = default;

			SemanticLidarDetection(
				float	 x,
				float	 y,
				float	 z,
				float	 intensity,
				uint32_t ring,
				uint32_t idx,
				uint32_t tag)
				: LidarDetection(x, y, z, intensity, ring)
				, object_idx{ idx }
				, object_tag{ tag } {}

			SemanticLidarDetection(
				FVector3d p,
				float	  intensity,
				uint32_t  ring,
				uint32_t  idx,
				uint32_t  tag)
				: LidarDetection(p, intensity, ring)
				, object_idx{ idx }
				, object_tag{ tag } {}

			void WritePlyHeaderInfo(std::ostream& out) const {
				out << "property float32 x\n"
					   "property float32 y\n"
					   "property float32 z\n"
					   "property float32 Intensity\n"
					   "property uint32 Ring\n"
					   "property uint32 ObjIdx\n"
					   "property uint32 ObjTag";
			}

			void WriteDetection(std::ostream& out) const {
				out << point.X << ' ' << point.Y << ' ' << point.Z << ' ' << intensity
					<< ' ' << ring << ' ' << object_idx << ' ' << object_tag;
			}

			~SemanticLidarDetection() = default;
		};

		class SemanticLidarData : public LidarData {

		public:
			explicit SemanticLidarData(uint32_t ChannelCount = 0u)
				: LidarData(ChannelCount) {}

			SemanticLidarData(
				float*	 buffer,
				uint32_t ChannelCount = 0u,
				uint32_t NumPointComponents = 7u)
				: LidarData(ChannelCount) {
				Buffer = buffer;
				this->NumPointComponents = NumPointComponents;
			}

			SemanticLidarData& operator=(SemanticLidarData&&) = default;

			~SemanticLidarData() = default;

			void ResetMemory(std::vector<uint32_t> points_per_channel) {
				// std::memset(_header.data() + Index::SIZE, 0, sizeof(uint32_t) *
				// GetChannelCount());

				uint32_t total_points = static_cast<uint32_t>(std::accumulate(
					points_per_channel.begin(), points_per_channel.end(), 0));

				points.clear();
				points.reserve(total_points * NumPointComponents);

				NumPoints = 1;
				Buffer[0] = static_cast<float>(total_points);
			}

			void WritePointSync(SemanticLidarDetection& detection) {

				points.emplace_back(detection.point.X);
				points.emplace_back(detection.point.Y);
				points.emplace_back(detection.point.Z);
				points.emplace_back(detection.intensity);
				points.emplace_back(static_cast<float>(detection.ring));
				points.emplace_back(static_cast<float>(detection.object_idx));
				points.emplace_back(static_cast<float>(detection.object_tag));

				Buffer[NumPointComponents * NumPoints + 0] = detection.point.X / 100;
				Buffer[NumPointComponents * NumPoints + 1] = -detection.point.Y
					/ 100; // Negate Y for Unreal's coordinate system, so we return in
						   // RH coordinate system
				Buffer[NumPointComponents * NumPoints + 2] = detection.point.Z / 100;
				Buffer[NumPointComponents * NumPoints + 3] = detection.intensity;
				Buffer[NumPointComponents * NumPoints + 4] =
					static_cast<float>(detection.ring);
				Buffer[NumPointComponents * NumPoints + 5] =
					static_cast<float>(detection.object_idx);
				Buffer[NumPointComponents * NumPoints + 6] =
					static_cast<float>(detection.object_tag);

				NumPoints++;
			}

		private:
			std::vector<float> points;
			float*			   Buffer;
			int				   NumPoints = 0;
		};
	} // namespace data

} // namespace holoocean
