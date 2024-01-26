//
// Created by lixin on 2023/12/19.
//

bool FeatureExtractionGrid(image_t *img_in, m_corners *corners, const uint8_t expect_feat_num, const uint8_t thresh, const uint8_t edge_x,
                           const uint8_t edge_y, float *avg_intensity, bool lower_thresh_when_not_enough, bool use_harris_check) {
    const float kLowerThresh = 0.75f * thresh;
    const float kMinFeatNum = 0.5f * expect_feat_num;
    const float kHarrisThresh = -1.f;
    const int img_w = img_in->w;
    const int img_h = img_in->h;

    const int grid_wh = ceil(sqrt(expect_feat_num));
    const int cell_w = (img_w - edge_x * 2 + grid_wh - 1) / grid_wh;
    const int cell_h = (img_h - edge_y * 2 + grid_wh - 1) / grid_wh;

    // extract feature points
    m_corners corners_max;
    m_corners_init(corners, grid_wh * grid_wh);
    m_corners_init(&corners_max, grid_wh * grid_wh);

    uint16_t count = 0;

    for (int gr = 0; gr < grid_wh; ++gr) {
        for (int gc = 0; gc < grid_wh; ++gc) {
            int row_st = edge_y + gr * cell_h;
            int row_ed = IM_MIN((row_st + cell_h), (img_h - edge_y));
            int col_st = edge_x + gc * cell_w;
            int col_ed = IM_MIN((col_st + cell_w), (img_w - edge_x));

            int max_mag = 0;
            int max_mag_x = 0;
            int max_mag_y = 0;

            for (int r = row_st; r < row_ed; ++r) {
                int8_t *row_ptr = IMAGE_COMPUTE_GRAYSCALE_INT_8_PIXEL_ROW_PTR(img_in, r);
                for (int c = col_st; c < col_ed; ++c) {
                    int mag = IMAGE_GET_GRAYSCALE_INT_8_PIXEL_FAST(row_ptr, c);
                    if (mag > max_mag) {
                        max_mag = mag;
                        max_mag_x = c;
                        max_mag_y = r;
                    }
                }
            }
            corners_max.points[count].x = (uint16_t)max_mag_x;
            corners_max.points[count].y = (uint16_t)max_mag_y;
            corners_max.points[count].score = (uint16_t)max_mag;
            count++;
        }
    }

    float score_sum = 0;
    count = 0;
    m_corner *corners_points = corners->points;
    m_corner *corners_max_points = corners_max.points;
    int corners_max_size = corners_max.size;

    for (int i = 0; i < corners_max_size; ++i) {
        m_corner *corner = &corners_max_points[i];
        if (corner->score > thresh) {
            score_sum += corner->score;
            corners_points[count++] = *corner;
        }
    }
    corners->size = count;

    // if corners.size() is far less than expect_feat_num, we need to add more corners
    if (corners->size < kMinFeatNum && lower_thresh_when_not_enough) {
        score_sum = 0;
        count = 0;

        for (int i = 0; i < corners_max.size; ++i) {
            m_corner *corner = &corners_max_points[i];
            if (corner->score > kLowerThresh) {
                score_sum += corner->score;
                corners_points[count++] = *corner;
            }
        }
        corners->size = count;
    }

    *avg_intensity = corners->size == 0 ? 0 : score_sum / corners->size;

    if (use_harris_check) {
        // reject feature with low harris score
        int sz = corners->size;
        int harris_rej = 0;

        for (int i = 0; i < sz; ++i) {
            m_corner *corner = &corners_points[i];
            float harris_score = calculate_harris_score(img_in, *corner);
            if (harris_score < kHarrisThresh) {
                corners_points[i] = corners_points[sz - 1];
                --sz;
                --i;
                ++harris_rej;
            }
        }
        printf("harris_rej %d", harris_rej);
    }

    return true;
}