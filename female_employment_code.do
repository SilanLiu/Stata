*Project
*Silan Liu

******work directory******
cd "D:\lsl\WUSTL\2025 Spring\Computational\project\v5"
******open log******
cap log close 
log using "project_log_v5.log", replace



******************************************************************************
***************************Define Poverty Line & Clean************************
******************************************************************************
* input poverty line
input year poverty_line
2024 15060
2023 14580
2022 13590
2021 12880
2020 12760
2019 12490
2018 12140
2017 12060
2016 11880
2015 11770
end

save poverty_line.dta, replace

******use data******
use cps_project_v3,clear

* Make sure year is numeric
destring year, replace

merge m:1 year using poverty_line.dta

save "project_merged.dta", replace

use project_merged, clear

* Drop top-coded income
replace inctot = . if inctot == 999999999

* Generate binary poverty status
gen poverty_alt = inctot < poverty_line
label define pov_lbl 1 "Below poverty line" 0 "Above poverty line"
label values poverty_alt pov_lbl

tab year poverty_line
tab poverty_alt

* year-month variable
gen year_month = year * 100 + month
keep if year_month >= 201501

* sort
sort year_month cpsidp

* drop ASEC oversample (cpsidp == 0)
drop if cpsidp == 0 
keep if age >= 18 & age <= 64


******************************************************************************
*****************************CREATE NEW VARIABLES*****************************
******************************************************************************
*Pandemic Period
gen post_pandemic = (year_month >= 202210)
label define post_lbl 0 "Pre-pandemic" 1 "Post-pandemic"
label values post_pandemic post_lbl

*Gender
gen female = .
replace female = 1 if sex == 2
replace female = 0 if sex == 1
label define sex_lbl 1 "Female" 0 "Male"
label values female sex_lbl
tab female

*Marital Status
gen marst_group = .
replace marst_group = 1 if marst == 1  // Married, spouse present
replace marst_group = 0 if inlist(marst, 2, 3, 4, 5, 6, 7)  // All other statuses

label define marst_lbl 1 "Married" 0 "Not Married"
label values marst_group marst_lbl
tab marst_group

*Race 
gen race_group = .
replace race_group = 1 if race == 100 // White
replace race_group = 2 if race == 200 // Black
replace race_group = 3 if race == 650 | race == 651 // Asian
replace race_group = 4 if  hispan >= 100 & hispan <= 612 
replace race_group = 5 if race >= 300 & race < 999 & !inlist(race,100,200,650,651) // Other

label define race_lbl 1 "White" 2 "Black" 3 "Asian" 4 "Hispanic" 5 "other"
label values race_group race_lbl
tab race_group

*Educ
gen educ_group = .
replace educ_group = 1 if educ == 0 | (educ >= 2 & educ < 73)  // Less than high school
replace educ_group = 2 if educ == 73 // High school graduate
replace educ_group = 3 if (educ >= 80 & educ <= 110) | (educ >= 120 & educ <= 122) // Some college
replace educ_group = 4 if educ == 111 | (educ >= 123 & educ <= 125) // College graduate

label define educ_lbl 1 "Less than high school" 2 "High school grad" 3 "Some college" 4 "College grad"
label values educ_group educ_lbl
tab educ_group

*Industry Classification
gen ind_group = .
replace ind_group = 1 if inrange(ind1990, 10, 60)
replace ind_group = 2 if inrange(ind1990, 100, 392)
replace ind_group = 3 if inrange(ind1990, 400, 472)
replace ind_group = 4 if inrange(ind1990, 500, 691)
replace ind_group = 5 if inrange(ind1990, 700, 791)
replace ind_group = 6 if inrange(ind1990, 800, 893)
replace ind_group = 7 if inrange(ind1990, 900, 932)
label define ind_lbl 1 "Primary Sector" 2 "Manufacturing" 3 "Transport & Utilities" 4 "Trade" 5 "Finance & Real Estate" 6 "Entertainment/Services" 7 "Public Sector"
label values ind_group ind_lbl
tab ind_group if female == 1
tab ind_group if female == 0

*Work Status
gen work_type = .
replace work_type = 1 if inrange(wkstat, 10, 15) // Full-time
replace work_type = 2 if inrange(wkstat, 20, 22) | inrange(wkstat, 40, 42)   // Part-time
replace work_type = 3 if inlist(wkstat, 50, 60) // Unemployed

label define work_lbl 1 "Full-time" 2 "Part-time" 3 "Unemployed" 
label values work_type work_lbl
tab work_type

*Hours Worked
replace uhrsworkt = . if uhrsworkt == 999

*Why Not Work //
gen whynotwork_cat = .
replace whynotwork_cat = 1 if whynwly == 1
replace whynotwork_cat = 2 if whynwly == 2
replace whynotwork_cat = 3 if whynwly == 3
replace whynotwork_cat = 4 if whynwly == 4
replace whynotwork_cat = 5 if inlist(whynwly,5,6,7)
label define why_lbl 1 "Job Market" 2 "Health" 3 "Family" 4 "Schooling" 5 "Other"
label values whynotwork_cat why_lbl
tab whynotwork_cat

* Save as a new file
save "project_cleaned_v5.dta", replace
******************************************************************************
*****************************Descriptive Statistics***************************
******************************************************************************
***use data
use project_cleaned_v5,clear
tab female educ_group
tab female ind_group
tab female race_group
tab female poverty_alt
tab work_type
******************************************************************************
*****************************Graph Analysis*****************************
******************************************************************************

*Employed dummy
gen employed = inlist(work_type, 1, 2)
label variable employed "Employed (1=Yes, 0=No)"

*Dummy for full-time and part-time work
gen full_time = (work_type == 1)
gen part_time = (work_type == 2)
label variable full_time "Full-time work (1=Yes, 0=No)"
label variable part_time "Part-time work (1=Yes, 0=No)"

* 
preserve
collapse (mean) full_time part_time [aweight=wtfinl], by(year female)
* Full-time rate by gender
twoway (line full_time year if female==0, lcolor(navy)) ///
       (line full_time year if female==1, lcolor(maroon)) ///
       , title("Full-time Rate by Gender") ///
         legend(order(1 "Male" 2 "Female")) ///
         ytitle("Mean Full-time") xtitle("Year") ///
         name(g1, replace)
graph export "1_gender_fulltime_comparison.jpg", replace

* Part-time rate by gender
twoway (line part_time year if female==0, lcolor(navy)) ///
       (line part_time year if female==1, lcolor(maroon)) ///
       , title("Part-time Rate by Gender") ///
         legend(order(1 "Male" 2 "Female")) ///
         ytitle("Mean Part-time") xtitle("Year") ///
         name(g2, replace)
graph export "2_gender_parttime_comparison.jpg", replace 		 
restore 

* Income
mean inctot if female==1 & employed==1, over(poverty_alt)

****** Education ******
preserve

* keep employed female
keep if female == 1 & employed == 1 & !missing(educ_group, post_pandemic, full_time, part_time)

* Collapse
collapse (mean) full_time part_time [aweight=wtfinl], by(educ_group post_pandemic)

* Full time x pre-pandemic
graph bar full_time if post_pandemic == 0, over(educ_group, label(angle(45))) ///
    bar(1, color(blue)) ///
    title("Female Full-time Rate by Education (Pre-pandemic)") ///
    ytitle("Full-time Rate") ///
    legend(off)
graph export "3a_female_fulltime_pre.jpg", width(1800) replace


* Full time x post-pandemic
graph bar full_time if post_pandemic == 1, over(educ_group, label(angle(45))) ///
    bar(1, color(blue)) ///
    title("Female Full-time Rate by Education (Post-pandemic)") ///
    ytitle("Full-time Rate") ///
    legend(off)
graph export "3b_female_fulltime_post.jpg", width(1800) replace


* Part-time x pre-pandemic
graph bar part_time if post_pandemic == 0, over(educ_group, label(angle(45))) ///
    bar(1, color(blue)) ///
    title("Female Part-time Rate by Education (Pre-pandemic)") ///
    ytitle("Part-time Rate") ///
    legend(off)
graph export "4a_female_parttime_pre.jpg", width(1800) replace

* === Part-time: Post-pandemic only ===
graph bar part_time if post_pandemic == 1, over(educ_group, label(angle(45))) ///
    bar(1, color(blue)) ///
    title("Female Part-time Rate by Education (Post-pandemic)") ///
    ytitle("Part-time Rate") ///
    legend(off)
graph export "4b_female_parttime_post.jpg", width(1800) replace

restore

****** Industry ******
preserve
collapse (mean) full_time part_time [aweight=wtfinl], ///
    by(year female ind_group)

* Industry
* 1.Female × Industry × Full-time
twoway ///
  (line full_time year if female==1 & ind_group==1, lcolor(blue)) ///
  (line full_time year if female==1 & ind_group==2, lcolor(red)) ///
  (line full_time year if female==1 & ind_group==3, lcolor(green)) ///
  (line full_time year if female==1 & ind_group==4, lcolor(orange)) ///
  (line full_time year if female==1 & ind_group==5, lcolor(purple)) ///
  (line full_time year if female==1 & ind_group==6, lcolor(cyan)) ///
  (line full_time year if female==1 & ind_group==7, lcolor(brown)) ///
, title("Female: Full-time Rate by Industry") ///
  legend(order(1 "Primary" 2 "Manufacturing" 3 "Transport" 4 "Trade" ///
  5 "Finance" 6 "Services" 7 "Public")) ///
    ytitle("Full-time Rate") xtitle("Year")
graph export "5_female_fulltime_industry.jpg", replace

* 2. Female × Industry × Part-time
twoway ///
  (line part_time year if female==1 & ind_group==1, lcolor(blue)) ///
  (line part_time year if female==1 & ind_group==2, lcolor(red)) ///
  (line part_time year if female==1 & ind_group==3, lcolor(green)) ///
  (line part_time year if female==1 & ind_group==4, lcolor(orange)) ///
  (line part_time year if female==1 & ind_group==5, lcolor(purple)) ///
  (line part_time year if female==1 & ind_group==6, lcolor(cyan)) ///
  (line part_time year if female==1 & ind_group==7, lcolor(brown)) ///
  , title("Female: Part-time Rate by Industry") ///
  legend(order(1 "Primary" 2 "Manufacturing" 3 "Transport" 4 "Trade" ///
  5 "Finance" 6 "Services" 7 "Public")) ///
  ytitle("Full-time Rate") xtitle("Year")
graph export "6_female_parttime_industry.jpg", replace

* 3. Male × Industry × Full-time
twoway ///
  (line full_time year if female==0 & ind_group==1, lcolor(blue)) ///
  (line full_time year if female==0 & ind_group==2, lcolor(red)) ///
  (line full_time year if female==0 & ind_group==3, lcolor(green)) ///
  (line full_time year if female==0 & ind_group==4, lcolor(orange)) ///
  (line full_time year if female==0 & ind_group==5, lcolor(purple)) ///
  (line full_time year if female==0 & ind_group==6, lcolor(cyan)) ///
  (line full_time year if female==0 & ind_group==7, lcolor(brown)) ///
  , title("Male: Full-time Rate by Industry") ///
  legend(order(1 "Primary" 2 "Manufacturing" 3 "Transport" 4 "Trade" ///
  5 "Finance" 6 "Services" 7 "Public")) ///
  ytitle("Full-time Rate") xtitle("Year")
graph export "7_male_fulltime_industry.jpg", replace

* 4. Male × Industry × Part-time
twoway ///
  (line part_time year if female==0 & ind_group==1, lcolor(blue)) ///
  (line part_time year if female==0 & ind_group==2, lcolor(red)) ///
  (line part_time year if female==0 & ind_group==3, lcolor(green)) ///
  (line part_time year if female==0 & ind_group==4, lcolor(orange)) ///
  (line part_time year if female==0 & ind_group==5, lcolor(purple)) ///
  (line part_time year if female==0 & ind_group==6, lcolor(cyan)) ///
  (line part_time year if female==0 & ind_group==7, lcolor(brown)) ///
  , title("Male: Part-time Rate by Industry") ///
  legend(order(1 "Primary" 2 "Manufacturing" 3 "Transport" 4 "Trade" ///
   5 "Finance" 6 "Services" 7 "Public")) ///
   ytitle("Full-time Rate") xtitle("Year")
graph export "8_male_parttime_industry.jpg", replace
restore


******************************************************************************
*****************************Regression Analysis*****************************
******************************************************************************


******Full-Time and Part-time ******
**Full-Time
logit full_time i.female##i.post_pandemic i.poverty_alt i.educ_group i.ind_group ///
      i.race_group i.marst_group age i.year ///
      if employed == 1 [pweight=wtfinl], vce(robust)
* Marginal effect: Female × Post-pandemic
margins female#post_pandemic
marginsplot, title("Probability of Full-time Work: Female × Post-pandemic")
graph export "9_female_fulltime_post_logit.jpg", replace

**Part-time
logit part_time i.female##i.post_pandemic i.poverty_alt i.educ_group i.ind_group ///
       i.race_group i.marst_group age i.year i.month ///
       if employed == 1 [pweight=wtfinl], vce(robust)
* Predicted probabilities
margins female#post_pandemic
marginsplot, title("Probability of Part-time Work: Female × Post-pandemic")
graph export "10_female_parttime_post.jpg", replace


****** Unemployed and Employed ******
logit employed i.female##i.post_pandemic ///
    i.educ_group i.ind_group i.race_group i.marst_group ///
    age i.year i.month [pweight=wtfinl], vce(robust)

margins female#post_pandemic
marginsplot, title("Probability of Employment: Female × Post-pandemic")
graph export "11_female_employment_post_logit.jpg", replace






