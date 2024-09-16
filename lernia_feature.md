---
title: "Motion"
author: Giovanni Marelli
date: 2019-11-01
rights:  Creative Commons Non-Commercial Share Alike 3.0
language: en-US
output: 
	md_document:
		variant: markdown_strict+backtick_code_blocks+autolink_bare_uris+markdown_github
---

# lernia features

Feature building from the lernia library

![lernia](f_lernia/lernia_logo.png "lernia logo")
_lernia library_

## data sources

We collect weather data from [darksky api](http://darksky.net) 

![data_sources](f_lernia/darksky.png "darksky")
_darksky weather api_

And census data from [eurostat](http://ec.europa.eu/eurostat/statistical-atlas/gis/viewer/?config=REF-GRID.json&mids=2,3,7&o=1,0.5,1&ch=4,5&center=55.2272,1.56222,4&)

![data_sources](f_lernia/census_data.png "census_data")
_eurostat census data_

## distribution

For each feature is important to undestand statistical properties such as:

* distribution of variance (normal, Poisson, multimodal...)
  * error/loss function
* periodicity 
  * transform function
* autocorrelation (entropy, information)
* noise kind
* decomposition (find relevant sub-components)


![stat_prob](f_lernia/stat_prop.png "statistical properties")
_statistical properties of a time series_


## normalization

We should evaluate the distribution of each variable, this view is confusing

![norm_no](f_lernia/norm_no.png "no norm")
_no norm_

An important operation for model convergence and performances is to normalize data. We can than see in one view all variances and skweness

![norm_minmax](f_lernia/norm_minmax.png "minmax norm")
_minmax norm_

Outliers can completly skew the distribution of variables and make learning difficult, we therefore remove the extreme percentiles 

![norm_5-95](f_lernia/norm_5-95.png "norm 5-95")
_norm 5-95_

We remove percentiles and normalize to one

![norm_1-99](f_lernia/norm_1-99.png "norm 1-99")
_norm 1-99_

Correlation between features is important to exclude features which are derivable

![feat_corr](f_lernia/feat_corr.png "feature correlation")
_feature correlation_

Outlier removal is not changing the correlation between features

![feat_corr](f_lernia/feat_corr_noNorm.png "feature correlation")
_feature correlation_

Some features have too many outliers, we decide to put a threshold and transform the feature into a logistic variable

![norm_cat](f_lernia/norm_cat.png "norm cat")
_norm cat_

Apart from boxplot is important to visualize data density and spot the multimodal distributions

![norm_joyplot](f_lernia/norm_joyplot.png "norm joyplot")
_norm joyplot_

We than sort the features and exclude highly skewed variables

![norm_logistic](f_lernia/norm_logistic.png "no logistic")
_norm logistic features_

## feature reduction

Looking at the 2d cross correlation we understand a lot about interaction between features

![dimension](f_lernia/selected_features.png "selected features")
_selected features_

And we can have a preliminary understanding about how features interacts

![feature_2dcor](f_lernia/weather_feature.png "feature 2d correlation")
_feature 2d correlation_

We know that apparent temperature is dependent from temperature, humidity, windSpeed, windBearing, cloudCover but we might not know why. Apparent temperature can be an important predictor so basically we can reduce the other components with a PCA

![pca](f_lernia/pca.png "pca")
_pca on derivate feature_

Interestingly the first component explains most of the feature set but doesn't explain the apparent temperature which is describes in the second component

![pca](f_lernia/pca_ratio.png "pca")
_components importance_

For the same components we can investigate other metrics

![feat_pairs](f_lernia/feat_pairs.png "feature pairs")
_feature pair metrics_

## replace

### replace NaNs

Working with python doesn't leave many options, contrary to R almost any library return errors. We therfore interpolate or drop lines.

![replace_nan](f_lernia/replace_nan.png "replace nan")
_replacing nans with interpolation_

The main issue with interpolation is at the boundary, special cases should be treated

## data cubes

If we have a lot of time series per location, or multiple signal superimposing we look at the chi square distribution to understand where outlier sequence windows are off

![chi](f_lernia/chisq_dis.png "chi square distribution")
_chi square distribution_

We than replace the off windows with a neighboring cluster

![replace](f_lernia/replace volatile.png "replace volatile")
_replace volatile sequences_

## feature importance

Feature importance is a function that simple models return. Since models don't agree on the same feature importance and production model will even come to much different conclusions.

![featImp_norm](f_lernia/featImp_normNo.png "feature iportance")
_feature importance no norm_

Normalization stabilize agreement between models

![featImp_norm](f_lernia/featImp_norm.png "feature iportance")
_feature importance norm_

We can apply as well a feature regularisation checking against a Lasso or a Ridge regression which features are relevant for the predicting variable

![feat_regularisation](f_lernia/feat_regularisation.png "feature regularisation")
_regularisation of features, mismatch in results depending on the regressor_

We than iterate model trainings removing one feature per time and calculate performaces. We can than understand how much is every feature important for the training

![feat_knock](f_lernia/feat_knockOut.png "feat knock out")
_feature knock out_

Strangely removing ozone and pressure the rain prediction suffers. We than analyze time series a realize a big gap in historical data and realize the few data where misleading for the model

![feat_time](f_lernia/feat_time.png "feature time series")
_feature time series_

## predictability

The meaning of building features is to achieve good predictability, if we want to predict rain we have differences in performace between models

![predictability](f_lernia/pred_normNo.png "predictability")
_predictability, no norm_

Cleaning features all the models perform basically the same

![predictability](f_lernia/pred_norm.png "predictability")
_predictability, normed_

Same if we train on spatial feature on a binned prediction variable

![predictability](f_lernia/predSpace_normNo.png "predictability")
_predictability, no norm_

After feature cleaning we have better agreement between models

![predictability](f_lernia/pred_norm.png "predictability")
_predictability, normed_

## transformation

### transform lines

Detailed information can be compressed fitting curves

![time_series](f_lernia/time_series.png "time series")
_simplify complexity_

For a many day time series we can distinguish periods from trends

![time_series](f_lernia/time_series_day.png "time series")
_simplify complexity_

### transform dimesionality

Time series can be transformed in pictures

![poly](f_lernia/prep_poly_smooth.png "series picture")
_time series in pictures_

Which is important to induce correlation between days and use more sofisticated methods

![ref_pred](f_lernia/ref_pred.png "reference prediction")
_reference prediction_

### transform interpolate

We can interpolate data to have more precise information and induce correlation between neighbors

![interpolate](f_lernia/popDens_interp.png "pop dens interpolate")
_interpolate population density_

### transform distribution

If we want to know how dense is an area with a particular geo feature

![spot building](f_lernia/spot_building.png "spot building")
_spot building distance_

We can reduce the density of feature fitting the radial histogram and returning the convexity of the parabola

![degeneracy](f_lernia/spatial_degeneracy.png "spatial degeneracy")
_spatial degeneracy, parabola convexity_

## boosting

If we apply boosting the distribution will change and therefore we can train another model to predict the residuals

![stat_prob](f_lernia/resid_distribution.png "statistical properties")
_residual distribution_
