*HW2
*Silan Liu

gl SA = 1 		//switch to SA = 0 to calculate unadjusted rates and export to Excel. SA = 1 to calculate seasonally adjusted rates and 
				//export both NSA and SA rates to the same Excel
******work directory******
cd "D:\lsl\WUSTL\2025 Spring\Computational\HW2"

******open log******
cap log close
log using "HW2_log.log", replace



******use data******
use cps_hw2,clear


*** year-month variable
gen year_month = ym(year, month)
format year_month %tm 

order month year_month, after(year)
keep if year >= 1976 & year <= 2021

******************************************************************
******************************** Q2 *****************************
******************************************************************

*** a. sample selection: Civilian labor force***
drop if empstat == 1   //drop military personnel 
drop if age < 16  //CPS defines civilian noninstitutional population to include those aged 16+


*** b. generate unemployed dummy***

* Generate the employed dummy
gen employed =1 if empstat == 10 | empstat == 12
replace employed = 0 if empstat >= 20 & empstat <= 22

lab def employed_lab 0 "0 - unemployed" 1 "1 - employed"
lab val employed employed_lab 
tab employed

*** c. sort the data by cpsidp and declare as panel***
sort year_month cpsidp
xtset cpsidp year_month

*** d. generate EU dummy***
gen EU_dummy = 1 if employed == 0 & L1.employed == 1
replace EU_dummy = 0 if employed ==1 & L1.employed == 1
lab def EU_dummy_lab 0 "0 - employed " 1 "1 - employed to unemployed"
lab val EU_dummy EU_dummy_lab
tab EU_dummy

*** e. generate UE dummy***
gen UE_dummy = 1 if employed == 1 & L1.employed == 0
replace UE_dummy = 0 if employed == 0 & L1.employed == 0
lab def UE_dummy_lab 0 "0 - unemployed " 1 "1 - unemployed to employed"
lab val UE_dummy UE_dummy_lab
tab UE_dummy

*** f. generate EE dummy***
gen EE_dummy = 1 if employed == 1 & empsame == 1
replace EE_dummy = 0 if employed == 1 & empsame == 2
lab def EE_dummy_lab 0 "0 - employed not change " 1 "1 - employed & change"
lab val EE_dummy EE_dummy_lab
tab EE_dummy

*** g. calculate EU, UE,EE rates***
***** Calculate Transition Rates *****
if $SA == 0 {
    * Unadjusted Rates Calculation
    collapse (mean) EU_rate_nsa = EU_dummy UE_rate_nsa = UE_dummy EE_rate_nsa = EE_dummy [aweight=wtfinl], by(year_month)
	
    
    * Export Unadjusted Rates to Excel
    export excel year_month EU_rate_nsa UE_rate_nsa EE_rate_nsa using "Transition_Rates.xlsx", firstrow(variable) keepcellfmt sheet("Unadjusted", replace)
}

if $SA == 1 {
    * Calculate Non-Seasonally Adjusted Rates
    collapse (mean) EU_rate_nsa = EU_dummy UE_rate_nsa = UE_dummy EE_rate_nsa = EE_dummy [aweight=wtfinl], by(year_month)

    * Extract Month for Seasonal Adjustment
    gen month = month(dofm(year_month))
    order month, after(year_month)

    * Seasonal Adjustment for EU
    egen EU_m = mean(EU_rate_nsa) if !missing(EU_rate_nsa)
    regress EU_rate_nsa b12.month
    predict EU_se
    gen EU_rate_sa = EU_rate_nsa - EU_se + EU_m

    * Seasonal Adjustment for UE
    egen UE_m = mean(UE_rate_nsa) if !missing(UE_rate_nsa)
    regress UE_rate_nsa b12.month
    predict UE_se
    gen UE_rate_sa = UE_rate_nsa - UE_se + UE_m

    * Seasonal Adjustment for EE
    egen EE_m = mean(EE_rate_nsa) if !missing(EE_rate_nsa)
    regress EE_rate_nsa b12.month
    predict EE_se
    gen EE_rate_sa = EE_rate_nsa - EE_se + EE_m

    * Export to Excel
    export excel year_month EU_rate_nsa EU_rate_sa UE_rate_nsa UE_rate_sa EE_rate_nsa EE_rate_sa using "Transition_Rates.xlsx", firstrow(variable) keepcellfmt sheet("Adjusted", modify)
}

*** j. graph ***

* EU Rate Plot
twoway (line EU_rate_nsa year_month, lcolor(blue)) ///
       (line EU_rate_sa year_month, lcolor(red)) ///
       , title("Employment to Unemployment (EU) Rate") ///
         xtitle("Year-Month") ///
         ytitle("EU Rate") ///
		 ylabel(0(.05)0.15, format(%9.2f)) ///
         legend(order(1 "NSA" 2 "SA")) ///
         graphregion(color(white))
graph save EU_plot, replace
graph export "EU_rate.jpg", replace

* UE Rate Plot
twoway (line UE_rate_nsa year_month, lcolor(blue)) ///
       (line UE_rate_sa year_month, lcolor(red)) ///
       , title("Unemployment to Employment (UE) Rate") ///
         xtitle("Year-Month") ///
         ytitle("UE Rate") ///
		 ylabel(0(.1)1, format(%9.2f)) ///
         legend(order(1 "NSA" 2 "SA")) ///
         graphregion(color(white))
graph save UE_plot, replace
graph export "UE_rate.jpg", replace

* EE Rate Plot
twoway (line EE_rate_nsa year_month, lcolor(blue)) ///
       (line EE_rate_sa year_month, lcolor(red)) ///
       , title("Employment to Employment (EE) Rate") ///
         xtitle("Year-Month") ///
         ytitle("EE Rate") ///
		 ylabel(0(.05)0.05, format(%9.2f)) ///
         legend(order(1 "NSA" 2 "SA")) ///
         graphregion(color(white))
graph save EE_plot, replace
graph export "EE_rate.jpg", replace

* Combine the three plots into a single large figure
graph combine EU_plot.gph UE_plot.gph EE_plot.gph, ///
              title("EU, UE, and EE Transition Rates Over Time") ///
              rows(2) cols(2) ///
			  iscale(0.7)

* Export the combined figure as a single JPG file
graph export "Transition_Rates_Combined.jpg", replace
