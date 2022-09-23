function [BW] = Binarizar_rojo(R, G, B)
%createMask  Threshold RGB image using auto-generated code from colorThresholder app.
%  [BW,MASKEDRGBIMAGE] = createMask(RGB) thresholds image RGB using
%  auto-generated code from the colorThresholder app. The colorspace and
%  range for each channel of the colorspace were set within the app. The
%  segmentation mask is returned in BW, and a composite of the mask and
%  original RGB images is returned in maskedRGBImage.

% Auto-generated by colorThresholder app on 24-Aug-2022
%------------------------------------------------------


% Convert RGB image to chosen color space
% I = RGB;

% Define thresholds for channel 1 based on histogram settings
channel1Min = 202.000;
channel1Max = 255.000;

% Define thresholds for channel 2 based on histogram settings
channel2Min = 0.000;
channel2Max = 19.000;

% Define thresholds for channel 3 based on histogram settings
channel3Min = 0.000;
channel3Max = 7.000;

% Create mask based on chosen histogram thresholds
sliderBW = (R >= channel1Min ) & (R <= channel1Max) & ...
    (G >= channel2Min ) & (G <= channel2Max) & ...
    (B >= channel3Min ) & (B <= channel3Max);
BW = sliderBW;

se = strel('disk',10);

BW = imclose(BW,se);


end