#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>

#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <chrono>
#include <iomanip>

namespace fs = std::filesystem;

std::string datasetPath;
int nfeatures;
float ratio;
double ransacThreshold;
fs::path resultsPath;

// ======================================================
// Результаты одного режима
// ======================================================

struct Result
{
    int keypoints = 0;
    int goodMatches = 0;
    int inliers = 0;

    double detectTime = 0;
    double matchTime = 0;
    double homographyTime = 0;
    double totalTime = 0;

    double cornerError = -1;
};


// ======================================================
// Находим JSON в
// ======================================================

fs::path findJson(const fs::path& photosPath)
{
    for (const auto& file :
        fs::directory_iterator(photosPath))
    {
        if (file.path().extension() == ".json")
            return file.path();
    }

    return {};
}


// ======================================================
// Читаем doc_quad для конкретной фотографии
// ======================================================

std::vector<cv::Point> loadDocQuad(
    const fs::path& jsonPath,
    const std::string& filename
)
{
    cv::FileStorage json(
        jsonPath.string(),
        cv::FileStorage::READ
    );

    cv::FileNode metadata =
        json["_via_img_metadata"];


    for (auto imageIt = metadata.begin();
        imageIt != metadata.end();
        ++imageIt)
    {
        cv::FileNode imageData = *imageIt;

        std::string currentFilename;
        imageData["filename"] >> currentFilename;

        if (currentFilename != filename)
            continue;


        cv::FileNode regions =
            imageData["regions"];


        for (auto regionIt = regions.begin();
            regionIt != regions.end();
            ++regionIt)
        {
            cv::FileNode region = *regionIt;

            std::string fieldName;

            region["region_attributes"]
                ["field_name"]
                >> fieldName;


            if (fieldName != "doc_quad")
                continue;


            std::vector<int> x;
            std::vector<int> y;

            region["shape_attributes"]
                ["all_points_x"]
                >> x;

            region["shape_attributes"]
                ["all_points_y"]
                >> y;


            return {
                {x[0], y[0]},
                {x[1], y[1]},
                {x[2], y[2]},
                {x[3], y[3]}
            };
        }
    }

    return {};
}


// ======================================================
// Средняя ошибка четырёх углов
// ======================================================

double calculateCornerError(
    const std::vector<cv::Point2f>& predicted,
    const std::vector<cv::Point>& real
)
{
    double error = 0;

    for (int i = 0; i < 4; ++i)
    {
        cv::Point2f realPoint(
            static_cast<float>(real[i].x),
            static_cast<float>(real[i].y)
        );

        error += cv::norm(
            predicted[i] - realPoint
        );
    }

    return error / 4.0;
}


// ======================================================
// FULL или ROI
// ======================================================

Result processMode(
    const cv::Mat& templateImage,
    const std::vector<cv::KeyPoint>& templateKeypoints,
    const cv::Mat& templateDescriptors,

    const cv::Mat& photoImage,
    const cv::Mat& mask,

    const std::vector<cv::Point>& docQuad,

    cv::Ptr<cv::ORB> orb,

    const std::string& outputName
)
{
    Result result;


    // ==================================================
    // ORB
    // ==================================================

    std::vector<cv::KeyPoint> photoKeypoints;
    cv::Mat photoDescriptors;

    auto start =
        std::chrono::steady_clock::now();


    orb->detectAndCompute(
        photoImage,
        mask,
        photoKeypoints,
        photoDescriptors
    );


    auto end =
        std::chrono::steady_clock::now();


    result.keypoints =
        static_cast<int>(
            photoKeypoints.size()
            );


    result.detectTime =
        std::chrono::duration<double, std::milli>(
            end - start
        ).count();


    // ==================================================
    // Matching
    // ==================================================

    cv::BFMatcher matcher(
        cv::NORM_HAMMING
    );

    std::vector<std::vector<cv::DMatch>>
        knnMatches;


    start =
        std::chrono::steady_clock::now();


    matcher.knnMatch(
        templateDescriptors,
        photoDescriptors,
        knnMatches,
        2
    );


    std::vector<cv::DMatch> goodMatches;



    for (const auto& matches : knnMatches)
    {
        if (matches.size() < 2)
            continue;


        if (
            matches[0].distance <
            ratio * matches[1].distance
            )
        {
            goodMatches.push_back(
                matches[0]
            );
        }
    }


    end =
        std::chrono::steady_clock::now();


    result.goodMatches =
        static_cast<int>(
            goodMatches.size()
            );


    result.matchTime =
        std::chrono::duration<double, std::milli>(
            end - start
        ).count();


    // ==================================================
    // Сохраняем matches
    // ==================================================

    cv::Mat matchesImage;

    cv::drawMatches(
        templateImage,
        templateKeypoints,
        photoImage,
        photoKeypoints,
        goodMatches,
        matchesImage
    );


    cv::imwrite(
        (resultsPath / (outputName + "_matches.jpg")).string(),
        matchesImage
    );


    // Для homography нужно минимум 4 совпадения
    if (goodMatches.size() < 4)
    {
        result.totalTime =
            result.detectTime +
            result.matchTime;

        return result;
    }


    // ==================================================
    // Точки для Homography
    // ==================================================

    std::vector<cv::Point2f> templatePoints;
    std::vector<cv::Point2f> photoPoints;


    for (const auto& match : goodMatches)
    {
        templatePoints.push_back(
            templateKeypoints[
                match.queryIdx
            ].pt
        );

        photoPoints.push_back(
            photoKeypoints[
                match.trainIdx
            ].pt
        );
    }


    // ==================================================
    // Homography
    // ==================================================

    cv::Mat inlierMask;


    start =
        std::chrono::steady_clock::now();


    cv::Mat H =
        cv::findHomography(
            templatePoints,
            photoPoints,
            cv::RANSAC,
            ransacThreshold,
            inlierMask
        );


    end =
        std::chrono::steady_clock::now();


    result.homographyTime =
        std::chrono::duration<double, std::milli>(
            end - start
        ).count();


    result.totalTime =
        result.detectTime +
        result.matchTime +
        result.homographyTime;


    if (H.empty())
        return result;


    result.inliers =
        cv::countNonZero(
            inlierMask
        );


    // ==================================================
    // Углы template
    // ==================================================

    std::vector<cv::Point2f> templateCorners =
    {
        {0, 0},

        {
            static_cast<float>(
                templateImage.cols
            ),
            0
        },

        {
            static_cast<float>(
                templateImage.cols
            ),

            static_cast<float>(
                templateImage.rows
            )
        },

        {
            0,

            static_cast<float>(
                templateImage.rows
            )
        }
    };


    // ==================================================
    // Переносим углы на фотографию
    // ==================================================

    std::vector<cv::Point2f> predictedCorners;


    cv::perspectiveTransform(
        templateCorners,
        predictedCorners,
        H
    );


    // ==================================================
    // Ошибка относительно doc_quad
    // ==================================================

    result.cornerError =
        calculateCornerError(
            predictedCorners,
            docQuad
        );


    // ==================================================
    // Визуализация локализации
    // ==================================================

    cv::Mat localizationImage;


    cv::cvtColor(
        photoImage,
        localizationImage,
        cv::COLOR_GRAY2BGR
    );


    // Зелёный = найденная homography
    for (int i = 0; i < 4; ++i)
    {
        cv::line(
            localizationImage,
            predictedCorners[i],
            predictedCorners[(i + 1) % 4],
            cv::Scalar(0, 255, 0),
            8
        );
    }


    // Красный = настоящий doc_quad
    for (int i = 0; i < 4; ++i)
    {
        cv::line(
            localizationImage,
            docQuad[i],
            docQuad[(i + 1) % 4],
            cv::Scalar(0, 0, 255),
            8
        );
    }


    cv::imwrite(
        (resultsPath / (outputName + "_localization.jpg")).string(),
        localizationImage
    );


    return result;
}


// ======================================================
// Вывод результата
// ======================================================

void printResult(
    const std::string& name,
    const Result& result
)
{
    std::cout
        << "\n"
        << name
        << "\n"

        << "Keypoints:      "
        << result.keypoints
        << "\n"

        << "Good matches:   "
        << result.goodMatches
        << "\n"

        << "Inliers:        "
        << result.inliers
        << "\n"

        << "Detect time:    "
        << result.detectTime
        << " ms\n"

        << "Match time:     "
        << result.matchTime
        << " ms\n"

        << "Homography:     "
        << result.homographyTime
        << " ms\n"

        << "Total time:     "
        << result.totalTime
        << " ms\n"

        << "Corner error:   "
        << result.cornerError
        << " px\n";
}

// ======================================================
// MAIN
// ======================================================

int main()
{
    // Чтение конфига
    cv::FileStorage config("config.yaml", cv::FileStorage::READ);

    config["dataset_path"] >> datasetPath;
    config["nfeatures"] >> nfeatures;
    config["ratio"] >> ratio;
    config["ransac_threshold"] >> ransacThreshold;

    config.release();

    std::cout
        << std::fixed
        << std::setprecision(2);

    // Создаём папку результатов
    fs::path datasetRoot = datasetPath;

    fs::path templatesPath = datasetRoot / "templates";
    fs::path photosPath = datasetRoot / "photo";
    resultsPath = datasetRoot / "results";

    fs::create_directories(resultsPath);

    // Находим JSON
    fs::path jsonPath =
        findJson(photosPath);

    // Для средних значений
    double fullTime = 0;
    double roiTime = 0;

    double fullMatches = 0;
    double roiMatches = 0;

    double fullInliers = 0;
    double roiInliers = 0;

    double fullError = 0;
    double roiError = 0;

    int count = 0;


    // ==================================================
    // Проходим по всем templates/*.jpg
    // ==================================================

    for (const auto& file :
        fs::directory_iterator(
            templatesPath
        ))
    {
        if (
            file.path().extension()
            != ".jpg"
            )
        {
            continue;
        }


        std::string filename =
            file.path()
            .filename()
            .string();


        std::string stem =
            file.path()
            .stem()
            .string();


        fs::path templatePath =
            file.path();


        fs::path photoPath =
            photosPath / filename;


        std::cout
            << "\n"
            << "============================\n"
            << "PHOTO: "
            << filename
            << "\n"
            << "============================\n";


        // ==================================================
        // Загружаем изображения
        // ==================================================

        cv::Mat templateImage =
            cv::imread(
                templatePath.string(),
                cv::IMREAD_GRAYSCALE
            );


        cv::Mat photoImage =
            cv::imread(
                photoPath.string(),
                cv::IMREAD_GRAYSCALE
            );


        // ==================================================
        // doc_quad
        // ==================================================

        std::vector<cv::Point> docQuad =
            loadDocQuad(
                jsonPath,
                filename
            );


        // ==================================================
        // ROI mask
        // ==================================================

        cv::Mat roiMask =
            cv::Mat::zeros(
                photoImage.size(),
                CV_8UC1
            );


        cv::fillConvexPoly(
            roiMask,
            docQuad,
            cv::Scalar(255)
        );


        cv::imwrite(
            (resultsPath / (stem + "_roi_mask.jpg")).string(),
            roiMask
        );


        // ==================================================
        // ORB на template
        // ==================================================

        auto orb =
            cv::ORB::create(nfeatures);


        std::vector<cv::KeyPoint>
            templateKeypoints;


        cv::Mat templateDescriptors;


        orb->detectAndCompute(
            templateImage,
            cv::noArray(),
            templateKeypoints,
            templateDescriptors
        );


        // ==================================================
        // FULL
        // ==================================================

        cv::Mat noMask;


        Result full =
            processMode(
                templateImage,
                templateKeypoints,
                templateDescriptors,

                photoImage,
                noMask,

                docQuad,
                orb,

                stem + "_full"
            );


        // ==================================================
        // ROI
        // ==================================================

        Result roi =
            processMode(
                templateImage,
                templateKeypoints,
                templateDescriptors,

                photoImage,
                roiMask,

                docQuad,
                orb,

                stem + "_roi"
            );


        // ==================================================
        // Вывод
        // ==================================================

        printResult(
            "FULL",
            full
        );


        printResult(
            "ROI",
            roi
        );

        // ==================================================
        // Средние
        // ==================================================

        fullTime += full.totalTime;
        roiTime += roi.totalTime;

        fullMatches += full.goodMatches;
        roiMatches += roi.goodMatches;

        fullInliers += full.inliers;
        roiInliers += roi.inliers;

        fullError += full.cornerError;
        roiError += roi.cornerError;

        ++count;
    }

    // ==================================================
    // Итог
    // ==================================================

    std::cout
        << "\n\n"
        << "========== AVERAGE ==========\n";


    std::cout
        << "\nFULL\n"

        << "Good matches: "
        << fullMatches / count
        << "\n"

        << "Inliers:      "
        << fullInliers / count
        << "\n"

        << "Time:         "
        << fullTime / count
        << " ms\n"

        << "Corner error: "
        << fullError / count
        << " px\n";


    std::cout
        << "\nROI\n"

        << "Good matches: "
        << roiMatches / count
        << "\n"

        << "Inliers:      "
        << roiInliers / count
        << "\n"

        << "Time:         "
        << roiTime / count
        << " ms\n"

        << "Corner error: "
        << roiError / count
        << " px\n";


    return 0;
}