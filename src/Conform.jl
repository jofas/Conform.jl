module Conform

  export CP, train!, predict

  export NCS

  export NCS1NearestNeighbor


  include("../RedBlackTree.jl/src/RedBlackTree.jl")

  using .RedBlackTree


  mutable struct CapacityArray{N} # {{{
    container::Array{Float64, N}
    capacity::Int64
    ptr::Int64
  end


  function CapacityArray{N}(capacity, columns...) where N # {{{
    container = Array{Float64, N}( undef, capacity
                                 , columns... )

    CapacityArray{N}( container
                    , capacity
                    , 1 )
  end # }}}


  function Base.push!( collection::CapacityArray{N} # ... {{{
                     , items... ) where N

    increase_if_full!(collection, length(items))

    place = ptr_non_growable_dims(collection)

    for item in items
      collection.container[collection.ptr, place...] = item
      collection.ptr += 1
    end
  end # }}}


  Base.size(self::CapacityArray{N}) where N =
    (self.ptr - 1, size(self.container)[2:end]...)

  Base.size(self::CapacityArray{N}, d) where N =
    d <= ndims(self.container) ? size(self)[d] : 1


  function Base.show(io::IO, ca::CapacityArray{N}) where N # {{{
    elems = elements(ca)

    if elems == nothing
      print(io, typeof(ca), ": Uninitialized")
    else
      print(io, typeof(ca), ": ", something(elems))
    end
  end # }}}


  Base.getindex(self::CapacityArray{N}, key...) where N =
    something(elements(self))[key...]


  Base.setindex!(self::CapacityArray{N}, v, key...) where N =
    something(elements(self))[key...] = v


  Base.firstindex(self::CapacityArray{N}) where N = 1
  Base.firstindex(self::CapacityArray{N}, d) where N = 1


  Base.lastindex(self::CapacityArray{N}) where N =
    *(size(self)...)

  Base.lastindex(self::CapacityArray{N}, d) where N =
    size(self, d)


  Base.axes(self::CapacityArray{N}) where N =
    Base.OneTo.(size(self))

  Base.axes(self::CapacityArray{N}, d) where N =
    Base.OneTo(size(self, d))


  function increase_if_full!( self::CapacityArray{N} # ... {{{
                            , amount_to_add::Int64 ) where N

    if size(self.container, 1) - self.ptr + 1 < amount_to_add
      increase!(self)
    end
  end # }}}


  function increase!(self::CapacityArray{N}) where N # {{{
    add_capacity = Array{Float64, N}( undef
                                    , self.capacity
                                    , size(self)[2:end]... )

    self.container = vcat(self.container, add_capacity)
  end # }}}


  function elements( self::CapacityArray{N} # ... {{{
                   )::Union{Some{SubArray}, Nothing} where N

    self.ptr > 1 ?
      Some(view( self.container, 1:self.ptr-1
               , ptr_non_growable_dims(self)... )) : nothing
  end # }}}


  function ptr_non_growable_dims( self::CapacityArray{N} # ... {{{
                                ) where N

    [Colon() for _ in 1:ndims(self.container) - 1]
  end # }}}

  # }}}


  abstract type NCS end


  struct NCS1NearestNeighbor <: NCS # {{{
    Δ::Function
    containers::Vector{CapacityArray{2}}
  end


  function NCS1NearestNeighbor( Δ::Function #... {{{
                              , dims_of_feature_space
                              , label_space_size
                              ; capacity = 64 )

    Δwrapper( x::AbstractArray{Float64, 1}
            , X::AbstractArray{Float64, 2}) = begin

      δmin = Inf

      for x̂ in eachrow(X)
        δ = Δ(x, x̂)
        if δ < δmin δmin = δ end
      end

      δmin
    end

    containers =
      Vector{CapacityArray{2}}(undef, label_space_size)

    for i in 1:label_space_size
      containers[i] =
        CapacityArray{2}(capacity, dims_of_feature_space)
    end

    NCS1NearestNeighbor(Δwrapper, containers)
  end # }}}


  Base.length(::NCS1NearestNeighbor) = 1

  Base.iterate(self::NCS1NearestNeighbor) = (self, nothing)
  Base.iterate(::NCS1NearestNeighbor, ::Nothing) = nothing


  function train!( self::NCS1NearestNeighbor # ... {{{
                , x::A where A <: AbstractArray{Float64, 1}
                , y::Int64 )::Float64

    σ = score(self, x, y)

    push!(self.containers[y], x)

    σ
  end # }}}


  function score( self::NCS1NearestNeighbor # ... {{{
                , x::A where A <: AbstractArray{Float64, 1}
                , y::Int64 )::Float64

    min_Δ_or_∞(X) = begin
      elems = elements(X)

      elems == nothing ?
        Inf : self.Δ(x, something(elems))
    end

    Δsame = min_Δ_or_∞(self.containers[y])

    Δother = min([
      min_Δ_or_∞(X) for X in self.containers[1:end .!= y]
    ]...)

    Δsame == Δother ? Inf : Δsame / Δother
  end # }}}


  # }}}


  struct CP # {{{
    ncs::N where N <: NCS
    K::Function
    label_space_size::Int64
    scores::Array{RBTree{Float64}}
  end


  function CP( ncs::N where N <: NCS # ... {{{
             , K::Function
             , label_space_size::Int64
             , taxonomy_space_size::Int64
             ; capacity = 64 )

    scores = Vector{RBTree{Float64}}( undef
                                    , taxonomy_space_size )

    for i in 1:taxonomy_space_size
      scores[i] = RBTree{Float64}()
    end

    CP(ncs, K, label_space_size, scores)
  end # }}}


  Base.length(::CP) = 1

  Base.iterate(self::CP) = (self, nothing)
  Base.iterate(::CP, ::Nothing) = nothing


  function train!( self::CP # ... {{{
                 , x::A where A <: AbstractArray{Float64,1}
                 , y::Int64 )
    k = self.K(x, y)

    σ = train!(self.ncs, x, y)

    insert!(self.scores[k], σ)
  end # }}}


  function train!( self::CP # ... {{{
                 , X::Matrix{Float64}
                 , Y::Vector{Int64} )

    for (x, y) in zip(eachrow(X), Y)
      train!(self, x, y)
    end
  end # }}}


  function predict( self::CP # ... {{{
                  , x::A where A<:AbstractArray{Float64,1}
                  )

    if !initialized(self) return (1, 1.0) end

    Ξ = Vector{Float64}(undef, self.label_space_size)

    for y in 1:self.label_space_size
      Ξ[y] = p_val(self, x, y)
    end

    yₘₐₓ, ϕ₁, ϕ₂ = 1, .0, .0

    for (y, ϕ) in enumerate(Ξ)
      if ϕ > ϕ₁
        yₘₐₓ, ϕ₁, ϕ₂ = y, ϕ, ϕ₁
      elseif ϕ > ϕ₂
        ϕ₂ = ϕ
      end
    end

    (yₘₐₓ, ϕ₂)
  end # }}}


  function predict(self::CP, X::Matrix{Float64}) # {{{
    predictions = Vector{Tuple{Int64, Float64}}( undef
                                               , size(X, 1) )

    for (i, x) in enumerate(eachrow(X))
      predictions[i] = predict(self, x)
    end

    predictions
  end # }}}


  function p_val( self::CP # ... {{{
                , x::A where A <: AbstractArray{Float64, 1}
                , y::Int64 )::Float64

    k = self.K(x, y)

    σ = score(self.ncs, x, y)

    geq(self.scores[k], σ) / insertions(self.scores[k])
  end # }}}


  initialized(self::CP) = all( t -> insertions(t) >= 1
                             , self.scores )
  # }}}
end
