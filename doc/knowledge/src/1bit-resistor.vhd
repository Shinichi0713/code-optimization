process(clk)
begin
    if rising_edge(clk) then
        q <= d;  -- クロックが来たら入力dをqに保持
    end if;
end process;

