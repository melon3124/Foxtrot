if not cadets.empty:
    st.markdown('<div class="centered">', unsafe_allow_html=True)
    cols_per_row = 4
    num_rows = (len(cadets) + cols_per_row - 1) // cols_per_row

    # Display buttons in a grid
    for i in range(num_rows):
        cols = st.columns(cols_per_row)
        for j in range(cols_per_row):
            idx = i * cols_per_row + j
            if idx < len(cadets):
                cadet_display_name = cadets.iloc[idx]["FULL NAME_DISPLAY"]
                cadet_cleaned_name = cadets.iloc[idx]["FULL NAME"]
                with cols[j]:
                    # Use a unique key for each button to avoid Streamlit warnings
                    if st.button(cadet_display_name, key=f"cadet_button_{cadet_cleaned_name}_{cls}"):
                        st.session_state.selected_cadet_display_name = cadet_display_name
                        st.session_state.selected_cadet_cleaned_name = cadet_cleaned_name
                        # Rerun to update the cadet details immediately
                        st.rerun() 
    st.markdown('</div>', unsafe_allow_html=True)

    # Retrieve selected cadet names from session state for display and comparison
    current_selected_cadet_display_name = st.session_state.selected_cadet_display_name
    current_selected_cadet_cleaned_name = st.session_state.selected_cadet_cleaned_name

    if current_selected_cadet_cleaned_name:
        # Get the row from demo_df using the cleaned name for lookup
        # This lookup is more robust as 'FULL NAME' is the cleaned version
        row = demo_df[demo_df["FULL NAME"] == current_selected_cadet_cleaned_name].iloc[0]

        st.markdown(f"## Showing details for: {current_selected_cadet_display_name}")

        t1, t2, t3, t4, t5 = st.tabs(["👤 Demographics", "🏃 PFT", "📚 Academics", "🪖 Military", "⚖ Conduct"])

        with t1:
            pic, info = st.columns([1, 2])
            with pic:
                # Use the display name for the image file path
                img = f"profile_pics/{current_selected_cadet_display_name}.jpg"
                st.image(img if os.path.exists(img) else "https://via.placeholder.com/400", width=350)
            with info:
                left, right = st.columns(2)
                # Filter out the internal 'FULL NAME' and 'FULL NAME_DISPLAY' columns
                details = {k: v for k, v in row.items() if k not in ["FULL NAME", "FULL NAME_DISPLAY", "CLASS"]}
                for idx, (k, v) in enumerate(details.items()):
                    (left if idx % 2 == 0 else right).write(f"**{k}:** {v}")

       with t2:
# -------------------- PFT GRADE SCALE (Hard-coded from SCALE_PFT) --------------------
            scale_data = {}
        
            # Add 1CL MALE PUSHUPS data manually
            reps = list(range(53, 101))  # 53 to 100 inclusive
            grades = [
                0, 7.7, 7.7, 7.8, 7.8, 7.9, 8, 8.1, 8.2, 8.3,
                8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 9, 9.1, 9.2, 9.3,
                9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 10, 10, 10, 10,
                10, 10, 10, 10, 10, 10, 10, 10
            ]
            event_key = ("1CL", "MALE", "PUSHUPS")
            scale_data[event_key] = {rep: grade for rep, grade in zip(reps, grades)}
        
            try:
                pft = sheet_df(f"{cls} PFT")
                if pft.empty:
                    st.info("No PFT data available for this class.")
                else:
                    # Standardize cadet name
                    pft["NAME_CLEANED"] = pft["NAME"].astype(str).apply(clean_cadet_name_for_comparison)
                    record = pft[pft["NAME_CLEANED"] == current_selected_cadet_cleaned_name]
        
                    if record.empty:
                        st.info("No PFT data for this cadet.")
                    else:
                        record = record.iloc[0]
                        gender = row.get("GENDER", "").strip().upper() or "MALE"
        
                        # Raw scores
                        push_raw = record.get("PUSHUPS", "-")
                        sit_raw  = record.get("SITUPS", "-")
                        flex_raw = record.get("PULLUPS/FLEX ARM HANG", "-")
                        run_raw  = record.get("3.2KM", "-")  # assume time in minutes
        
                        # Convert raw input if possible, else None
                        push_grade = get_grade(cls, gender, "PUSHUPS", push_raw)
                        sit_grade  = get_grade(cls, gender, "SITUPS", sit_raw)
                        flex_grade = get_grade(cls, gender, "FLEX", flex_raw)
                        run_grade  = get_grade(cls, gender, "RUN", run_raw)
        
                        # Build DataFrame
                        table = pd.DataFrame([
                            {"Event": "Pushups", "Raw": push_raw, "Grade": push_grade,
                             "Interpretation": interpret_grade(push_grade)},
                            {"Event": "Situps", "Raw": sit_raw, "Grade": sit_grade,
                             "Interpretation": interpret_grade(sit_grade)},
                            {"Event": "Flex / Pull‑ups", "Raw": flex_raw, "Grade": flex_grade,
                             "Interpretation": interpret_grade(flex_grade)},
                            {"Event": "3.2 km Run (min)", "Raw": run_raw, "Grade": run_grade,
                             "Interpretation": interpret_grade(run_grade)},
                        ])
        
                        st.markdown("### PFT Breakdown")
                        st.dataframe(table, hide_index=True)
            except Exception as e:
                st.error(f"PFT load error: {e}")

             
        with t3: # Academics tab - main focus of the fix
            try:
                acad_sheet_map = {
                    "1CL": "1CL ACAD",
                    "2CL": "2CL ACAD",
                    "3CL": "3CL ACAD"
                }
                acad = sheet_df(acad_sheet_map[cls]) # Already cleaned by sheet_df

                if acad.empty:
                    st.info("No Academic data available for this class.")
                else:
                    # Consistently look for "NAME" column in academic sheets
                    target_name_col = "NAME" 

                    if target_name_col not in acad.columns:
                        st.error(f"Error: Expected column '{target_name_col}' not found in the academic sheet '{acad_sheet_map[cls]}'.")
                        st.write(f"Available columns in '{acad_sheet_map[cls]}': {acad.columns.tolist()}")
                        # No return here, just display error and skip remaining logic for this tab
                    else: # Only proceed if the target_name_col exists
                        # Apply robust cleaning to the academic sheet's name column
                        acad['NAME_CLEANED'] = acad[target_name_col].astype(str).apply(clean_cadet_name_for_comparison)

                        r = acad[acad["NAME_CLEANED"] == current_selected_cadet_cleaned_name]

                        if not r.empty:
                            r = r.iloc[0]
                            # Drop the original name column and the cleaned name column before displaying
                            # Ensure 'NAME' is dropped if it's the target column for consistency
                            df_data = r.drop([col for col in r.index if col in [target_name_col, 'NAME_CLEANED']], errors='ignore')
                            
                            df = pd.DataFrame({"Subject": df_data.index, "Grade": df_data.values})
                            
                            df["Grade_Numeric"] = pd.to_numeric(df["Grade"], errors='coerce')
                            df["Status"] = df["Grade_Numeric"].apply(lambda g: "Proficient" if g >= 7 else "Deficient" if pd.notna(g) else "N/A")
                            
                            st.dataframe(df[['Subject', 'Grade', 'Status']], hide_index=True)
                        else:
                            st.warning(f"No academic record found for {current_selected_cadet_display_name}.")
                            # Optional: For debugging, uncomment to see available cleaned names
                            # st.write("Available names in sheet (cleaned):", acad['NAME_CLEANED'].tolist())
            except Exception as e:
                st.error(f"Academic load error: {e}")
                # import traceback
                # st.error(traceback.format_exc()) # Uncomment for full traceback

        with t4:
            try:
                mil = sheet_df(f"{cls} MIL")
                if mil.empty:
                    st.info("No Military data available for this class.")
                else:
                    mil['NAME_CLEANED'] = mil["NAME"].astype(str).apply(clean_cadet_name_for_comparison)
                    r = mil[mil["NAME_CLEANED"] == current_selected_cadet_cleaned_name]

                    if not r.empty:
                        # Drop 'NAME' and 'NAME_CLEANED' from items before iterating
                        for subj, grade in r.iloc[0].drop(['NAME', 'NAME_CLEANED'], errors='ignore').items():
                            st.write(f"**{subj}:** {grade}")
                    else:
                        st.info("No Military data available for this cadet.")
            except Exception as e:
                st.error(f"Military load error: {e}")
                # import traceback
                # st.error(traceback.format_exc())

        with t5:
            try:
                cond = sheet_df("CONDUCT")
                if cond.empty:
                    st.info("No Conduct data available.")
                else:
                    cond['NAME_CLEANED'] = cond["NAME"].astype(str).apply(clean_cadet_name_for_comparison)
                    r = cond[cond["NAME_CLEANED"] == current_selected_cadet_cleaned_name]

                    if not r.empty:
                        # Drop 'NAME' and 'NAME_CLEANED' from items before iterating
                        for subj, val in r.iloc[0].drop(['NAME', 'NAME_CLEANED'], errors='ignore').items():
                            st.write(f"**{subj}:** {val}")
                    else:
                        st.info("No Conduct data available for this cadet.")
            except Exception as e:
                st.error(f"Conduct load error: {e}")
                # import traceback
                # st.error(traceback.format_exc())
else:
    st.warning(f"No cadets found for class {cls}.")
