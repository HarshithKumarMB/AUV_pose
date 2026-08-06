// Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#pragma once

#include <cstdint>
#include <vector>
#include <numeric>

namespace holoocean {
	namespace data {
		class LidarDetection {
		public:
			FVector3d point{};
			float	  intensity;
			uint32_t  ring{};

			LidarDetection() : point(0.0, 0.0, 0.0), intensity{ 0.0 }, ring{ 0 } {
				// Default constructor
			}

			LidarDetection(double x, double y, double z, float intensity, uint32_t ring)
				: point(x, y, z), intensity{ intensity }, ring{ ring } {}

			LidarDetection(FVector3d p, float intensity, uint32_t ring)
				: point(p), intensity{ intensity }, ring{ ring } {}

			virtual void WritePlyHeaderInfo(std::ostream& out) const {
				out << "property float32 x\n"
					   "property float32 y\n"
					   "property float32 z\n"
					   "property float32 I\n"
					   "property uint32 Ring";
			}

			virtual void WriteDetection(std::ostream& out) const {
				out << point.X << ' ' << point.Y << ' ' << point.Z << ' ' << intensity
					<< ' ' << ring;
			}

			virtual ~LidarDetection() = default;
		};

		class LidarData {

			static_assert(sizeof(float) == sizeof(uint32_t), "Invalid float size");

		public:
			explicit LidarData(uint32_t ChannelCount = 0u)
				: _header(Index::SIZE + ChannelCount, 0u) {
				_header[Index::ChannelCount] = ChannelCount;
			}

			LidarData(
				float*	 buffer,
				uint32_t ChannelCount = 0u,
				uint32_t NumPointComponents = 5u) {
				Buffer = buffer;
				channelCount = ChannelCount;
				this->NumPointComponents = NumPointComponents;
			}

			float GetHorizontalAngle() const { return horizontalAngle; }

			void SetHorizontalAngle(float angle) { this->horizontalAngle = angle; }

			uint32_t GetChannelCount() const { return channelCount; }

			std::vector<float> GetPoints() const {
				return std::vector<float>(points.begin(), points.end());
			}

			virtual void WriteChannelCount(std::vector<uint32_t> points_per_channel) {
				for (auto idxChannel = 0u; idxChannel < GetChannelCount(); ++idxChannel)
					_header[Index::SIZE + idxChannel] = points_per_channel[idxChannel];
			}

			virtual void ResetMemory(std::vector<uint32_t> points_per_channel) {
				// std::memset(_header.data() + Index::SIZE, 0, sizeof(uint32_t) *
				// GetChannelCount());

				uint32_t total_points = static_cast<uint32_t>(std::accumulate(
					points_per_channel.begin(), points_per_channel.end(), 0));

				points.clear();
				points.reserve(total_points * NumPointComponents);

				NumPoints = 1;
				Buffer[0] = static_cast<float>(total_points);
			}

			void WritePointSync(LidarDetection& detection) {

				points.emplace_back(detection.point.X);
				points.emplace_back(detection.point.Y);
				points.emplace_back(detection.point.Z);
				points.emplace_back(detection.intensity);
				points.emplace_back(static_cast<float>(detection.ring));

				Buffer[NumPointComponents * NumPoints + 0] = detection.point.X / 100;
				Buffer[NumPointComponents * NumPoints + 1] = -detection.point.Y
					/ 100; // Negate Y for Unreal's coordinate system, so we return in
						   // RH coordinate system
				Buffer[NumPointComponents * NumPoints + 2] = detection.point.Z / 100;
				Buffer[NumPointComponents * NumPoints + 3] = detection.intensity;
				Buffer[NumPointComponents * NumPoints + 4] =
					static_cast<float>(detection.ring);

				NumPoints++;
			}

			LidarData& operator=(LidarData&&) = default;

			virtual ~LidarData() = default;

			std::size_t CountOfPoints() { return points.size(); }

			std::vector<float> points;
			float*			   Buffer;
			int				   NumPoints = 0;
			float			   horizontalAngle = 0.0f;
			uint32_t		   channelCount = 0;

		protected:
			enum Index : size_t {
				HorizontalAngle,
				ChannelCount,
				SIZE
			};

			std::vector<uint32_t> _header = std::vector<uint32_t>(Index::SIZE, 0u);
			uint32_t			  NumPointComponents =
				5u; // Default to 5 components: x, y, z, intensity, ring
		};
	} // namespace data
} // namespace holoocean
