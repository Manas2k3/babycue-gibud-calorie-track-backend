# 📱 Flutter Integration Guide for Food Recognition & USDA Nutrition API

This guide provides everything required to integrate the **Food Recognition & USDA Nutrition API** into your Flutter application (iOS, Android, Web).

---

## 🛠️ 1. Required Packages

Add `http` and `image_picker` to your `pubspec.yaml`:

```yaml
dependencies:
  flutter:
    sdk: flutter
  http: ^1.2.0
  image_picker: ^1.1.0
```

---

## 📦 2. Dart Data Models

Create a file `lib/models/food_detection_model.dart`:

```dart
import 'dart:convert';

class DetectionResponse {
  final bool success;
  final String runId;
  final ImageDimensions imageSize;
  final List<DetectedFoodItem> detections;
  final List<DetectedFoodItem> top5;
  final TotalNutritionalSummary nutritionSummary;
  final Map<String, String> urls;

  DetectionResponse({
    required this.success,
    required this.runId,
    required this.imageSize,
    required this.detections,
    required this.top5,
    required this.nutritionSummary,
    required this.urls,
  });

  factory DetectionResponse.fromJson(Map<String, dynamic> json) {
    return DetectionResponse(
      success: json['success'] ?? false,
      runId: json['run_id'] ?? '',
      imageSize: ImageDimensions.fromJson(json['image_size'] ?? {}),
      detections: (json['detections'] as List? ?? [])
          .map((item) => DetectedFoodItem.fromJson(item))
          .toList(),
      top5: (json['top5'] as List? ?? [])
          .map((item) => DetectedFoodItem.fromJson(item))
          .toList(),
      nutritionSummary: TotalNutritionalSummary.fromJson(json['nutrition_summary'] ?? {}),
      urls: Map<String, String>.from(json['urls'] ?? {}),
    );
  }
}

class ImageDimensions {
  final int width;
  final int height;

  ImageDimensions({required this.width, required this.height});

  factory ImageDimensions.fromJson(Map<String, dynamic> json) {
    return ImageDimensions(
      width: json['width'] ?? 0,
      height: json['height'] ?? 0,
    );
  }
}

class DetectedFoodItem {
  final String className;
  final double confidence;
  final List<double>? boxXyxyNorm; // [x_min, y_min, x_max, y_max] normalized 0.0-1.0
  final String source;
  final NutritionProfile? nutrition;

  DetectedFoodItem({
    required this.className,
    required this.confidence,
    this.boxXyxyNorm,
    required this.source,
    this.nutrition,
  });

  factory DetectedFoodItem.fromJson(Map<String, dynamic> json) {
    return DetectedFoodItem(
      className: json['class_name'] ?? 'Unknown',
      confidence: (json['confidence'] as num?)?.toDouble() ?? 0.0,
      boxXyxyNorm: json['box_xyxy_norm'] != null
          ? (json['box_xyxy_norm'] as List).map((e) => (e as num).toDouble()).toList()
          : null,
      source: json['source'] ?? 'yolo_confirmed',
      nutrition: json['nutrition_per_100g'] != null
          ? NutritionProfile.fromJson(json['nutrition_per_100g'])
          : null,
    );
  }
}

class NutritionProfile {
  final double? caloriesKcal;
  final double? proteinG;
  final double? carbsG;
  final double? fatG;
  final double? fiberG;
  final double? sugarsG;
  final double? sodiumMg;
  final double? potassiumMg;
  final double? calciumMg;
  final double? ironMg;
  final double? vitaminAIu;
  final double? vitaminCMg;
  final double? vitaminDIu;
  final double servingSize;
  final String servingSizeUnit;
  final String? dbSource;
  final String? matchedName;

  NutritionProfile({
    this.caloriesKcal,
    this.proteinG,
    this.carbsG,
    this.fatG,
    this.fiberG,
    this.sugarsG,
    this.sodiumMg,
    this.potassiumMg,
    this.calciumMg,
    this.ironMg,
    this.vitaminAIu,
    this.vitaminCMg,
    this.vitaminDIu,
    required this.servingSize,
    required this.servingSizeUnit,
    this.dbSource,
    this.matchedName,
  });

  factory NutritionProfile.fromJson(Map<String, dynamic> json) {
    return NutritionProfile(
      caloriesKcal: (json['calories_kcal'] as num?)?.toDouble(),
      proteinG: (json['protein_g'] as num?)?.toDouble(),
      carbsG: (json['carbs_g'] as num?)?.toDouble(),
      fatG: (json['fat_g'] as num?)?.toDouble(),
      fiberG: (json['fiber_g'] as num?)?.toDouble(),
      sugarsG: (json['sugars_g'] as num?)?.toDouble(),
      sodiumMg: (json['sodium_mg'] as num?)?.toDouble(),
      potassiumMg: (json['potassium_mg'] as num?)?.toDouble(),
      calciumMg: (json['calcium_mg'] as num?)?.toDouble(),
      ironMg: (json['iron_mg'] as num?)?.toDouble(),
      vitaminAIu: (json['vitamin_a_iu'] as num?)?.toDouble(),
      vitaminCMg: (json['vitamin_c_mg'] as num?)?.toDouble(),
      vitaminDIu: (json['vitamin_d_iu'] as num?)?.toDouble(),
      servingSize: (json['serving_size'] as num?)?.toDouble() ?? 100.0,
      servingSizeUnit: json['serving_size_unit'] ?? 'g',
      dbSource: json['source'],
      matchedName: json['matched_name'],
    );
  }
}

class TotalNutritionalSummary {
  final int totalItemsDetected;
  final double totalCaloriesKcal;
  final double totalProteinG;
  final double totalCarbsG;
  final double totalFatG;
  final double totalFiberG;
  final double totalSugarsG;
  final double totalSodiumMg;

  TotalNutritionalSummary({
    required this.totalItemsDetected,
    required this.totalCaloriesKcal,
    required this.totalProteinG,
    required this.totalCarbsG,
    required this.totalFatG,
    required this.totalFiberG,
    required this.totalSugarsG,
    required this.totalSodiumMg,
  });

  factory TotalNutritionalSummary.fromJson(Map<String, dynamic> json) {
    return TotalNutritionalSummary(
      totalItemsDetected: json['total_items_detected'] ?? 0,
      totalCaloriesKcal: (json['total_calories_kcal'] as num?)?.toDouble() ?? 0.0,
      totalProteinG: (json['total_protein_g'] as num?)?.toDouble() ?? 0.0,
      totalCarbsG: (json['total_carbs_g'] as num?)?.toDouble() ?? 0.0,
      totalFatG: (json['total_fat_g'] as num?)?.toDouble() ?? 0.0,
      totalFiberG: (json['total_fiber_g'] as num?)?.toDouble() ?? 0.0,
      totalSugarsG: (json['total_sugars_g'] as num?)?.toDouble() ?? 0.0,
      totalSodiumMg: (json['total_sodium_mg'] as num?)?.toDouble() ?? 0.0,
    );
  }
}
```

---

## 📡 3. Flutter API Service

Create `lib/services/food_recognition_service.dart`:

```dart
import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;
import '../models/food_detection_model.dart';

class FoodRecognitionService {
  // Replace with your Google Cloud Run service URL after deployment
  static const String baseUrl = 'https://food-recognition-api-gibud-f7cc9-uc.a.run.app';

  /// Sends image file to /predict endpoint and parses detection + nutrition result
  static Future<DetectionResponse> predictFoodImage(File imageFile, {bool saveVisuals = false}) async {
    final uri = Uri.parse('$baseUrl/predict?save_visuals=$saveVisuals');
    
    final request = http.MultipartRequest('POST', uri);
    request.files.add(await http.MultipartFile.fromPath('file', imageFile.path));
    
    final streamedResponse = await request.send().timeout(const Duration(seconds: 30));
    final response = await http.Response.fromStream(streamedResponse);

    if (response.statusCode == 200) {
      final Map<String, dynamic> jsonData = jsonDecode(response.body);
      return DetectionResponse.fromJson(jsonData);
    } else {
      throw Exception('Failed food detection API call (${response.statusCode}): ${response.body}');
    }
  }

  /// Health Check Endpoint
  static Future<bool> checkHealth() async {
    try {
      final response = await http.get(Uri.parse('$baseUrl/health'));
      return response.statusCode == 200;
    } catch (_) {
      return false;
    }
  }
}
```

---

## 🎨 4. Flutter Custom Bounding Box Canvas Painter

Create `lib/widgets/food_bounding_box_overlay.dart`:

```dart
import 'package:flutter/material.dart';
import '../models/food_detection_model.dart';

class FoodBoundingBoxOverlay extends StatelessWidget {
  final List<DetectedFoodItem> detections;
  final Size renderSize;

  const FoodBoundingBoxOverlay({
    Key? key,
    required this.detections,
    required this.renderSize,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      size: renderSize,
      painter: _BoundingBoxPainter(detections),
    );
  }
}

class _BoundingBoxPainter extends CustomPainter {
  final List<DetectedFoodItem> detections;

  _BoundingBoxPainter(this.detections);

  @override
  void paint(Canvas canvas, Size size) {
    final boxPaint = Paint()
      ..color = const Color(0xFFFF6D00)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.0;

    final fillPaint = Paint()
      ..color = const Color(0xFFFF6D00).withOpacity(0.2)
      ..style = PaintingStyle.fill;

    for (var item in detections) {
      if (item.boxXyxyNorm == null || item.boxXyxyNorm!.length < 4) continue;

      final norm = item.boxXyxyNorm!;
      final left = norm[0] * size.width;
      final top = norm[1] * size.height;
      final right = norm[2] * size.width;
      final bottom = norm[3] * size.height;

      final rect = Rect.fromLTRB(left, top, right, bottom);
      canvas.drawRect(rect, fillPaint);
      canvas.drawRect(rect, boxPaint);

      // Draw Label Badge
      final textSpan = TextSpan(
        text: '${item.className} (${(item.confidence * 100).toStringAsFixed(0)}%)',
        style: const TextStyle(
          color: Colors.white,
          fontSize: 12,
          fontWeight: FontWeight.bold,
        ),
      );

      final textPainter = TextPainter(
        text: textSpan,
        textDirection: TextDirection.ltr,
      )..layout();

      final badgeRect = Rect.fromLTWH(
        left,
        top - 20 < 0 ? top : top - 20,
        textPainter.width + 8,
        20,
      );

      canvas.drawRect(badgeRect, Paint()..color = const Color(0xFFFF6D00));
      textPainter.paint(canvas, Offset(badgeRect.left + 4, badgeRect.top + 2));
    }
  }

  @override
  bool shouldRepaint(covariant _BoundingBoxPainter oldDelegate) => true;
}
```
